from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from .config import AppConfig

try:  # Optional dependency
    from PIL import Image, ExifTags
except Exception:  # pragma: no cover - optional
    Image = None
    ExifTags = None

try:  # Optional dependency
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional
    PdfReader = None


Logger = Callable[[str], None]
ProgressEvent = Dict[str, int | str]


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".xml",
    ".ini",
    ".cfg",
    ".rst",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic"}


@dataclass
class PlannedMove:
    src: str
    dest: str


@dataclass
class Candidate:
    path: Path
    name: str
    root_dir: Path
    embedding: Optional[List[float]] = None
    metadata: Dict[str, str] | None = None
    content_snippet: str = ""


class CacheStore:
    def __init__(self, cache_dir: Path, model: str, logger: Logger):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"embeddings_{self._safe_model_name(model)}.json"
        self.logger = logger
        self._cache: Dict[str, List[float]] = {}
        self._loaded = False

    def _safe_model_name(self, model: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", model)

    def load(self) -> None:
        if self._loaded:
            return
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._cache = {
                        k: v for k, v in raw.items() if isinstance(v, list)
                    }
                self.logger(f"Loaded embedding cache: {len(self._cache)} entries")
            except Exception as e:
                self.logger(f"Cache load failed, continuing without cache: {e}")
        self._loaded = True

    def get(self, key: str) -> Optional[List[float]]:
        self.load()
        return self._cache.get(key)

    def set(self, key: str, embedding: List[float]) -> None:
        self.load()
        self._cache[key] = embedding

    def save(self) -> None:
        if not self._loaded:
            return
        tmp_path = self.cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        tmp_path.replace(self.cache_path)


class LocalClassifier:
    def __init__(self, name_rules: Dict[str, str], path_rules: Dict[str, str]):
        self.name_rules = {k.lower(): v for k, v in name_rules.items()}
        self.path_rules = {k.lower(): v for k, v in path_rules.items()}

    def classify(self, filename: str, path: str) -> Optional[str]:
        name = filename.lower()
        lower_path = path.lower()

        for k, v in self.path_rules.items():
            if k in lower_path:
                return v

        for k, v in self.name_rules.items():
            if k in name:
                return v

        return None


class MetadataExtractor:
    def __init__(self, cfg: AppConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger

    def extract(self, path: Path) -> Tuple[Dict[str, str], str]:
        metadata: Dict[str, str] = {}
        content_snippet = ""

        try:
            stat = path.stat()
        except Exception as e:
            self.logger(f"SKIP unreadable file: {path} ({e})")
            return metadata, content_snippet

        metadata["size_bytes"] = str(stat.st_size)
        metadata["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        metadata["extension"] = path.suffix.lower()

        if stat.st_size > self.cfg.max_file_bytes:
            self.logger(f"Large file, using metadata only: {path}")
            return metadata, content_snippet

        if self.cfg.use_pdf_meta and path.suffix.lower() == ".pdf":
            self._add_pdf_metadata(path, metadata)

        if self.cfg.use_exif and path.suffix.lower() in IMAGE_EXTENSIONS:
            self._add_exif_metadata(path, metadata)

        if path.suffix.lower() in TEXT_EXTENSIONS:
            content_snippet = read_text(path, self.cfg.max_text_chars)

        return metadata, content_snippet

    def _add_pdf_metadata(self, path: Path, metadata: Dict[str, str]) -> None:
        if PdfReader is None:
            self.logger("PDF metadata requested but pypdf is not installed.")
            return
        try:
            reader = PdfReader(str(path))
            info = reader.metadata
            if info:
                for key in ("title", "author", "subject"):
                    value = getattr(info, key, None)
                    if value:
                        metadata[f"pdf_{key}"] = str(value)
        except Exception as e:
            self.logger(f"PDF metadata error for {path}: {e}")

    def _add_exif_metadata(self, path: Path, metadata: Dict[str, str]) -> None:
        if Image is None:
            self.logger("EXIF requested but Pillow is not installed.")
            return
        try:
            with Image.open(path) as img:
                exif = img.getexif()
            if not exif:
                return
            tag_map = ExifTags.TAGS if ExifTags else {}
            for tag_id, value in exif.items():
                tag = tag_map.get(tag_id, str(tag_id))
                if tag in {"Make", "Model", "DateTime", "DateTimeOriginal"}:
                    metadata[f"exif_{tag}"] = str(value)
        except Exception as e:
            self.logger(f"EXIF metadata error for {path}: {e}")


class EmbeddingService:
    def __init__(
        self,
        client: OpenAI,
        cfg: AppConfig,
        logger: Logger,
        progress: Optional[Callable[[ProgressEvent], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self.progress = progress or (lambda _: None)
        self.should_stop = should_stop or (lambda: False)
        self.cache = CacheStore(Path(cfg.cache_dir), cfg.embedding_model, logger)

    def _cache_key(self, path: Path, size: int, mtime: float) -> str:
        raw = f"{path}|{size}|{mtime}|{self.cfg.embedding_model}|{self.cfg.max_text_chars}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_embeddings(
        self, items: List[Candidate], texts: List[str]
    ) -> List[Optional[List[float]]]:
        if len(items) != len(texts):
            raise ValueError("Items and texts must have same length")

        cached: List[Optional[List[float]]] = [None] * len(items)
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        for i, (item, text) in enumerate(zip(items, texts)):
            try:
                stat = item.path.stat()
            except Exception:
                continue
            key = self._cache_key(item.path, stat.st_size, stat.st_mtime)
            hit = self.cache.get(key)
            if hit is not None:
                cached[i] = hit
            else:
                missing_texts.append(text)
                missing_indices.append(i)

        if missing_texts:
            self.progress({"type": "embeddings_total", "value": len(missing_texts)})
            self.logger(f"Embedding batch size: {len(missing_texts)}")
            batch_size = 96
            processed = 0
            for start in range(0, len(missing_texts), batch_size):
                if self.should_stop():
                    self.logger("Cancel requested during embeddings.")
                    break
                batch = missing_texts[start : start + batch_size]
                response = self.client.embeddings.create(
                    model=self.cfg.embedding_model,
                    input=batch,
                )
                for offset, emb in enumerate(response.data):
                    idx = missing_indices[start + offset]
                    cached[idx] = emb.embedding
                    try:
                        stat = items[idx].path.stat()
                        key = self._cache_key(items[idx].path, stat.st_size, stat.st_mtime)
                        self.cache.set(key, emb.embedding)
                    except Exception:
                        pass
                processed += len(batch)
                self.progress({"type": "embeddings_done", "value": processed})

        self.cache.save()
        return cached


class Clusterer:
    def __init__(self, threshold: float, max_size: int, method: str, logger: Logger):
        self.threshold = threshold
        self.max_size = max_size
        self.method = method
        self.logger = logger

    def cluster(self, items: List[Candidate]) -> List[List[Candidate]]:
        if not items:
            return []
        if self.method != "centroid":
            self.logger(f"Unknown cluster method '{self.method}', using centroid.")

        clusters: List[Dict[str, object]] = []
        for item in items:
            if item.embedding is None:
                continue
            best_idx = -1
            best_sim = -1.0
            for idx, cluster in enumerate(clusters):
                members: List[Candidate] = cluster["items"]  # type: ignore[assignment]
                if len(members) >= self.max_size:
                    continue
                centroid: List[float] = cluster["centroid"]  # type: ignore[assignment]
                sim = cosine(item.embedding, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_idx >= 0 and best_sim >= self.threshold:
                clusters[best_idx]["items"].append(item)  # type: ignore[index]
                clusters[best_idx]["centroid"] = centroid_mean(
                    clusters[best_idx]["items"]  # type: ignore[index]
                )
            else:
                clusters.append({"items": [item], "centroid": item.embedding})

        return [cluster["items"] for cluster in clusters]  # type: ignore[list-item]


class Labeler:
    def __init__(self, client: OpenAI, cfg: AppConfig, logger: Logger):
        self.client = client
        self.cfg = cfg
        self.logger = logger

    def choose_category(self, files: List[Candidate]) -> str:
        names = [f.name for f in files]
        prompt = (
            "Pick the single best category from this list.\n"
            f"Categories: {json.dumps(self.cfg.taxonomy)}\n"
            "Return ONLY the category string, exactly as listed.\n"
            f"Files: {json.dumps(names)}"
        )
        response = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": "You are a precise file organizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.cfg.labeler_temperature,
        )
        raw = response.choices[0].message.content.strip()
        for option in self.cfg.taxonomy:
            if raw.lower() == option.lower():
                return option
        self.logger(f"Invalid category '{raw}', falling back to Unsorted")
        return "Unsorted"

    def choose_subfolder(self, files: List[Candidate], category: str) -> str:
        names = [f.name for f in files][:30]
        prompt = (
            "Create a short subfolder name under the given category.\n"
            "Return ONLY a short name, no slashes.\n"
            f"Category: {category}\n"
            f"Files: {json.dumps(names)}"
        )
        response = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": "You are a precise file organizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.cfg.labeler_temperature,
        )
        raw = response.choices[0].message.content.strip()
        return sanitize_segment(raw, default="General")


class Organizer:
    def __init__(
        self,
        cfg: AppConfig,
        logger: Optional[Logger] = None,
        progress: Optional[Callable[[ProgressEvent], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.logger = logger or (lambda _: None)
        self.progress = progress or (lambda _: None)
        self.should_stop = should_stop or (lambda: False)
        self.client = OpenAI()

        self.local_classifier = LocalClassifier(cfg.local_name_rules, cfg.local_path_rules)
        self.metadata_extractor = MetadataExtractor(cfg, self.log)
        self.embedding_service = EmbeddingService(
            self.client,
            cfg,
            self.log,
            progress=self.progress,
            should_stop=self.should_stop,
        )
        self.clusterer = Clusterer(
            threshold=cfg.embedding_threshold,
            max_size=cfg.cluster_max_size,
            method=cfg.cluster_method,
            logger=self.log,
        )
        self.labeler = Labeler(self.client, cfg, self.log)

    def log(self, msg: str) -> None:
        self.logger(msg)

    def plan(self) -> List[PlannedMove]:
        planned: List[PlannedMove] = []
        candidates: List[Candidate] = []
        scanned_files = 0

        self.log("Phase 1: Local smart organization")
        self.progress({"type": "scan_start", "value": 0})

        for root_dir in self.cfg.target_dirs:
            root_path = Path(root_dir)
            dest_root = root_path / self.cfg.dest_name
            dest_root.mkdir(parents=True, exist_ok=True)

            if self.cfg.include_subfolders:
                walker = os.walk(root_dir)
            else:
                walker = [(root_dir, [], os.listdir(root_dir))]

            for dirpath, dirnames, files in walker:
                if self._check_cancel("scan"):
                    return planned
                self._filter_dirs(dirnames)
                if self.cfg.dest_name in Path(dirpath).parts:
                    continue

                for filename in files:
                    if self._check_cancel("scan"):
                        return planned
                    if self.cfg.skip_hidden and filename.startswith("."):
                        continue
                    src = Path(dirpath) / filename
                    if src.is_dir():
                        continue
                    guess = self.local_classifier.classify(filename, dirpath)

                    if guess:
                        dest = dest_root / guess / filename
                        planned.append(PlannedMove(src=str(src), dest=str(dest)))
                    else:
                        if not self._eligible_for_ai(src):
                            continue
                        candidates.append(
                            Candidate(
                                path=src,
                                name=filename,
                                root_dir=root_path,
                            )
                        )
                    scanned_files += 1
                    if scanned_files % 200 == 0:
                        self.progress({"type": "scan_tick", "value": scanned_files})

        self.progress({"type": "scan_done", "value": scanned_files})

        self.log(f"Remaining files for AI: {len(candidates)}")
        if not candidates:
            return planned

        self.log("Phase 2: Embedding + Clustering")

        texts: List[str] = []
        for candidate in candidates:
            metadata, snippet = self.metadata_extractor.extract(candidate.path)
            candidate.metadata = metadata
            candidate.content_snippet = snippet
            texts.append(build_embedding_text(candidate, self.cfg.max_text_chars))

        embeddings = self.embedding_service.get_embeddings(candidates, texts)
        filtered_candidates: List[Candidate] = []
        for candidate, emb in zip(candidates, embeddings):
            if emb is None:
                self.log(f"SKIP embedding failed: {candidate.path}")
                continue
            candidate.embedding = emb
            filtered_candidates.append(candidate)

        clusters = self.clusterer.cluster(filtered_candidates)
        self.log(f"AI groups to classify: {len(clusters)}")
        self.progress({"type": "label_total", "value": len(clusters)})

        total_cost_est = estimate_cost(len(filtered_candidates), len(clusters))
        if total_cost_est > self.cfg.budget_limit:
            self.log(
                f"Budget would exceed (${total_cost_est:.2f} > ${self.cfg.budget_limit:.2f}). Stopping."
            )
            return planned

        self.log("Phase 3: GPT labeling")

        labeled = 0
        for cluster in clusters:
            if self._check_cancel("labeling"):
                return planned
            category = self.labeler.choose_category(cluster)
            subfolder = self.labeler.choose_subfolder(cluster, category)

            safe_category = sanitize_segment(category, default="Unsorted")
            safe_subfolder = sanitize_segment(subfolder, default="General")
            label_path = f"{safe_category}/{safe_subfolder}"

            for item in cluster:
                dest = item.root_dir / self.cfg.dest_name / label_path / item.name
                planned.append(PlannedMove(src=str(item.path), dest=str(dest)))
            labeled += 1
            self.progress({"type": "label_done", "value": labeled})

        return planned

    def apply(self, planned: List[PlannedMove]) -> None:
        for move in planned:
            if self._check_cancel("apply"):
                return
            src = Path(move.src)
            dest = Path(move.dest)
            if not src.exists():
                self.log(f"SKIP missing: {src}")
                continue

            dest = resolve_collision(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)

            if not self._is_within_dest_root(dest):
                self.log(f"SKIP unsafe destination: {dest}")
                continue

            shutil.move(str(src), str(dest))
            self.log(f"MOVE {src} -> {dest}")

    def _eligible_for_ai(self, path: Path) -> bool:
        ext = path.suffix.lower().lstrip(".")
        if self.cfg.skip_extensions and ext in {e.lower().lstrip(".") for e in self.cfg.skip_extensions}:
            self.log(f"SKIP extension excluded: {path}")
            return False
        if self.cfg.allowed_extensions:
            allowed = {e.lstrip(".").lower() for e in self.cfg.allowed_extensions}
            if ext not in allowed:
                self.log(f"SKIP extension not allowed: {path}")
                return False
        return True

    def _is_within_dest_root(self, dest: Path) -> bool:
        for root_dir in self.cfg.target_dirs:
            root_path = Path(root_dir) / self.cfg.dest_name
            try:
                if dest.resolve().is_relative_to(root_path.resolve()):
                    return True
            except Exception:
                continue
        return False

    def _filter_dirs(self, dirnames: List[str]) -> None:
        skip_set = {d.lower() for d in self.cfg.skip_dirs}
        kept = []
        for name in dirnames:
            lower = name.lower()
            if self.cfg.skip_hidden and name.startswith("."):
                continue
            if lower in skip_set or lower.endswith(".app"):
                continue
            kept.append(name)
        dirnames[:] = kept

    def _check_cancel(self, phase: str) -> bool:
        if self.should_stop():
            self.log(f"Cancel requested. Stopping during {phase}.")
            return True
        return False


def read_text(path: Path, max_chars: int) -> str:
    try:
        with path.open("r", errors="ignore") as f:
            return f.read()[:max_chars]
    except Exception:
        return ""


def cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def centroid_mean(items: List[Candidate]) -> List[float]:
    vectors = [item.embedding for item in items if item.embedding is not None]
    if not vectors:
        return []
    return list(np.mean(np.array(vectors), axis=0))


def sanitize_segment(raw: str, default: str) -> str:
    cleaned = raw.strip().replace("/", " ").replace("\\", " ")
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or cleaned in {".", ".."}:
        return default
    return cleaned[:60]


def build_embedding_text(candidate: Candidate, max_chars: int) -> str:
    parts = [f"filename: {candidate.name}"]
    if candidate.metadata:
        for key, value in candidate.metadata.items():
            parts.append(f"{key}: {value}")
    if candidate.content_snippet:
        parts.append("content:")
        parts.append(candidate.content_snippet)
    text = "\n".join(parts)
    return text[:max_chars]


def estimate_cost(num_embeddings: int, num_clusters: int) -> float:
    # Rough placeholder estimate; adjust if pricing changes.
    return (num_embeddings * 0.0001) + (num_clusters * 0.002)


def resolve_collision(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1
