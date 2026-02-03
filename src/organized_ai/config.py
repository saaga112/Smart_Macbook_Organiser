from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class AppConfig:
    target_dirs: List[str]
    dest_name: str
    budget_limit: float
    embedding_model: str
    chat_model: str
    embedding_threshold: float
    max_text_chars: int
    dry_run: bool
    cache_dir: str
    taxonomy: List[str]
    max_file_bytes: int
    allowed_extensions: List[str]
    skip_dirs: List[str]
    skip_hidden: bool
    skip_extensions: List[str]
    use_exif: bool
    use_pdf_meta: bool
    use_ocr: bool
    cluster_method: str
    cluster_max_size: int
    labeler_temperature: float
    local_name_rules: Dict[str, str]
    local_path_rules: Dict[str, str]
    include_subfolders: bool


def default_config() -> AppConfig:
    home = Path.home()
    return AppConfig(
        target_dirs=[str(home / "Downloads"), str(home / "Documents")],
        dest_name="Organized_AI",
        budget_limit=0.80,
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4.1-mini",
        embedding_threshold=0.88,
        max_text_chars=2000,
        dry_run=True,
        cache_dir=str(Path("project/.cache/organized_ai")),
        taxonomy=[
            "Documents",
            "Financial",
            "Bills",
            "Invoices",
            "Receipts",
            "Medical",
            "Legal",
            "Work",
            "Career",
            "Projects",
            "Education",
            "Personal",
            "Travel",
            "Photos",
            "Screenshots",
            "Music",
            "Videos",
            "Archives",
            "Unsorted",
        ],
        max_file_bytes=10_000_000,
        allowed_extensions=[],
        skip_dirs=[
            ".git",
            ".venv",
            ".terraform",
            "node_modules",
            ".app",
            "__pycache__",
            ".cache",
        ],
        skip_hidden=True,
        skip_extensions=[
            "dng",
            "zip",
            "7z",
            "rar",
            "tar",
            "gz",
            "tgz",
            "bz2",
            "xz",
            "jar",
            "iso",
            "dmg",
            "pkg",
            "exe",
            "mov",
            "mp4",
            "mkv",
            "avi",
        ],
        use_exif=False,
        use_pdf_meta=False,
        use_ocr=False,
        cluster_method="centroid",
        cluster_max_size=50,
        labeler_temperature=0.2,
        local_name_rules={
            "screenshot": "Screenshots",
            "invoice": "Documents/Invoices",
            "bill": "Documents/Bills",
            "tax": "Documents/Tax",
            "insurance": "Documents/Insurance",
            "medical": "Documents/Medical",
            "report": "Documents/Reports",
            "bank": "Documents/Bank",
            "resume": "Career/Resume",
            "cv": "Career/Resume",
            "offer": "Career/Offers",
            "agreement": "Legal/Agreements",
        },
        local_path_rules={
            "whatsapp": "Photos/WhatsApp",
            "telegram": "Photos/Telegram",
            "iphone": "Photos/iPhone",
        },
        include_subfolders=False,
    )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return default_config()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = default_config()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


def save_config(cfg: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    data = asdict(cfg)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
