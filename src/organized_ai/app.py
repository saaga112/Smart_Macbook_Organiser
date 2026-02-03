from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from dotenv import load_dotenv

from .config import AppConfig, load_config, save_config
from .organizer import Organizer


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Organized AI")
        self.root.geometry("900x600")

        load_dotenv()
        self.cfg = load_config()

        self.log_queue: queue.Queue[object] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self._build_ui()
        self._reset_progress()
        self._pump_logs()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_container = tk.Frame(frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = tk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        right = tk.Frame(frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Target dirs
        tk.Label(left, text="Target folders").pack(anchor="w")
        self.targets_list = tk.Listbox(left, height=8, width=50)
        self.targets_list.pack(fill=tk.X)

        for d in self.cfg.target_dirs:
            self.targets_list.insert(tk.END, d)

        btn_row = tk.Frame(left)
        btn_row.pack(fill=tk.X, pady=4)

        tk.Button(btn_row, text="Add", command=self._add_dir).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Remove", command=self._remove_dir).pack(side=tk.LEFT, padx=4)

        # Options
        tk.Label(left, text="Destination folder name").pack(anchor="w", pady=(10, 0))
        self.dest_name_var = tk.StringVar(value=self.cfg.dest_name)
        tk.Entry(left, textvariable=self.dest_name_var, width=50).pack(fill=tk.X)

        tk.Label(left, text="Budget limit (USD)").pack(anchor="w", pady=(10, 0))
        self.budget_var = tk.StringVar(value=str(self.cfg.budget_limit))
        tk.Entry(left, textvariable=self.budget_var, width=20).pack(fill=tk.X)

        tk.Label(left, text="Embedding similarity threshold").pack(anchor="w", pady=(10, 0))
        self.threshold_var = tk.StringVar(value=str(self.cfg.embedding_threshold))
        tk.Entry(left, textvariable=self.threshold_var, width=20).pack(fill=tk.X)

        tk.Label(left, text="Embedding model").pack(anchor="w", pady=(10, 0))
        self.embedding_model_var = tk.StringVar(value=self.cfg.embedding_model)
        tk.Entry(left, textvariable=self.embedding_model_var, width=50).pack(fill=tk.X)

        tk.Label(left, text="Chat model").pack(anchor="w", pady=(10, 0))
        self.chat_model_var = tk.StringVar(value=self.cfg.chat_model)
        tk.Entry(left, textvariable=self.chat_model_var, width=50).pack(fill=tk.X)

        tk.Label(left, text="Max text chars per file").pack(anchor="w", pady=(10, 0))
        self.max_chars_var = tk.StringVar(value=str(self.cfg.max_text_chars))
        tk.Entry(left, textvariable=self.max_chars_var, width=20).pack(fill=tk.X)

        self.dry_run_var = tk.BooleanVar(value=self.cfg.dry_run)
        tk.Checkbutton(left, text="Dry run (no moves)", variable=self.dry_run_var).pack(anchor="w", pady=10)

        self.advanced_visible = False
        self.advanced_btn = tk.Button(left, text="Advanced ▸", command=self._toggle_advanced)
        self.advanced_btn.pack(anchor="w", pady=(0, 6))

        self.advanced_frame = tk.Frame(left)

        self.include_subfolders_var = tk.BooleanVar(value=self.cfg.include_subfolders)
        tk.Checkbutton(
            self.advanced_frame,
            text="Include subfolders (recursive)",
            variable=self.include_subfolders_var,
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(self.advanced_frame, text="Cache directory").pack(anchor="w")
        self.cache_dir_var = tk.StringVar(value=self.cfg.cache_dir)
        tk.Entry(self.advanced_frame, textvariable=self.cache_dir_var, width=50).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Max file bytes (metadata only above)").pack(
            anchor="w", pady=(10, 0)
        )
        self.max_file_bytes_var = tk.StringVar(value=str(self.cfg.max_file_bytes))
        tk.Entry(self.advanced_frame, textvariable=self.max_file_bytes_var, width=20).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Allowed extensions (comma-separated)").pack(
            anchor="w", pady=(10, 0)
        )
        self.allowed_ext_var = tk.StringVar(value=",".join(self.cfg.allowed_extensions))
        tk.Entry(self.advanced_frame, textvariable=self.allowed_ext_var, width=50).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Skip directories (comma-separated)").pack(
            anchor="w", pady=(10, 0)
        )
        self.skip_dirs_var = tk.StringVar(value=",".join(self.cfg.skip_dirs))
        tk.Entry(self.advanced_frame, textvariable=self.skip_dirs_var, width=50).pack(fill=tk.X)

        self.skip_hidden_var = tk.BooleanVar(value=self.cfg.skip_hidden)
        tk.Checkbutton(self.advanced_frame, text="Skip hidden files/dirs", variable=self.skip_hidden_var).pack(
            anchor="w", pady=(6, 0)
        )

        tk.Label(self.advanced_frame, text="Skip extensions (comma-separated)").pack(
            anchor="w", pady=(10, 0)
        )
        self.skip_ext_var = tk.StringVar(value=",".join(self.cfg.skip_extensions))
        tk.Entry(self.advanced_frame, textvariable=self.skip_ext_var, width=50).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Taxonomy (comma-separated)").pack(
            anchor="w", pady=(10, 0)
        )
        self.taxonomy_var = tk.StringVar(value=",".join(self.cfg.taxonomy))
        tk.Entry(self.advanced_frame, textvariable=self.taxonomy_var, width=50).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Cluster method").pack(anchor="w", pady=(10, 0))
        self.cluster_method_var = tk.StringVar(value=self.cfg.cluster_method)
        tk.Entry(self.advanced_frame, textvariable=self.cluster_method_var, width=20).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Cluster max size").pack(anchor="w", pady=(10, 0))
        self.cluster_max_size_var = tk.StringVar(value=str(self.cfg.cluster_max_size))
        tk.Entry(self.advanced_frame, textvariable=self.cluster_max_size_var, width=10).pack(fill=tk.X)

        tk.Label(self.advanced_frame, text="Labeler temperature").pack(anchor="w", pady=(10, 0))
        self.labeler_temp_var = tk.StringVar(value=str(self.cfg.labeler_temperature))
        tk.Entry(self.advanced_frame, textvariable=self.labeler_temp_var, width=10).pack(fill=tk.X)

        self.use_exif_var = tk.BooleanVar(value=self.cfg.use_exif)
        tk.Checkbutton(self.advanced_frame, text="Use EXIF (images)", variable=self.use_exif_var).pack(
            anchor="w", pady=(10, 0)
        )

        self.use_pdf_meta_var = tk.BooleanVar(value=self.cfg.use_pdf_meta)
        tk.Checkbutton(
            self.advanced_frame, text="Use PDF metadata", variable=self.use_pdf_meta_var
        ).pack(anchor="w", pady=(4, 0))

        self.use_ocr_var = tk.BooleanVar(value=self.cfg.use_ocr)
        tk.Checkbutton(
            self.advanced_frame, text="Use OCR (placeholder)", variable=self.use_ocr_var
        ).pack(anchor="w", pady=(4, 0))

        action_row = tk.Frame(left)
        action_row.pack(fill=tk.X, pady=8)

        self.save_btn = tk.Button(action_row, text="Save Config", command=self._save_config)
        self.save_btn.pack(side=tk.LEFT)

        self.dry_run_btn = tk.Button(action_row, text="Run Dry", command=lambda: self._run(dry_run=True))
        self.dry_run_btn.pack(side=tk.LEFT, padx=6)

        self.apply_btn = tk.Button(action_row, text="Apply", command=lambda: self._run(dry_run=False))
        self.apply_btn.pack(side=tk.LEFT)

        self.cancel_btn = tk.Button(action_row, text="Cancel", command=self._cancel_run, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=6)

        # Progress
        progress_frame = tk.Frame(right)
        progress_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(progress_frame, text="Scan progress").pack(anchor="w")
        self.scan_progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.scan_progress.pack(fill=tk.X)
        self.scan_status = tk.Label(progress_frame, text="Scanned: 0 files")
        self.scan_status.pack(anchor="w", pady=(2, 6))

        tk.Label(progress_frame, text="AI progress").pack(anchor="w")
        self.ai_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.ai_progress.pack(fill=tk.X)
        self.ai_status = tk.Label(progress_frame, text="Embeddings: 0/0 | Labels: 0/0")
        self.ai_status.pack(anchor="w", pady=(2, 0))

        # Logs
        tk.Label(right, text="Logs").pack(anchor="w")
        log_frame = tk.Frame(right)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_left_configure(_: object) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        def _on_left_mousewheel(event: tk.Event) -> None:
            delta = -1 * int(event.delta / 120)
            left_canvas.yview_scroll(delta, "units")

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_left_configure)
        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

    def _add_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.targets_list.insert(tk.END, path)

    def _remove_dir(self) -> None:
        selected = list(self.targets_list.curselection())
        for idx in reversed(selected):
            self.targets_list.delete(idx)

    def _save_config(self) -> None:
        cfg = self._gather_config()
        save_config(cfg)
        self.log("Saved config.yaml")

    def _gather_config(self) -> AppConfig:
        target_dirs = list(self.targets_list.get(0, tk.END))
        allowed_extensions = [
            ext.strip().lstrip(".").lower()
            for ext in self.allowed_ext_var.get().split(",")
            if ext.strip()
        ]
        skip_dirs = [d.strip() for d in self.skip_dirs_var.get().split(",") if d.strip()]
        skip_extensions = [
            ext.strip().lstrip(".").lower()
            for ext in self.skip_ext_var.get().split(",")
            if ext.strip()
        ]
        taxonomy = [t.strip() for t in self.taxonomy_var.get().split(",") if t.strip()]
        if not taxonomy:
            taxonomy = self.cfg.taxonomy
        return AppConfig(
            target_dirs=target_dirs,
            dest_name=self.dest_name_var.get().strip() or "Organized_AI",
            budget_limit=float(self.budget_var.get()),
            embedding_model=self.embedding_model_var.get().strip() or "text-embedding-3-small",
            chat_model=self.chat_model_var.get().strip() or "gpt-4.1-mini",
            embedding_threshold=float(self.threshold_var.get()),
            max_text_chars=int(self.max_chars_var.get()),
            dry_run=self.dry_run_var.get(),
            cache_dir=self.cache_dir_var.get().strip() or self.cfg.cache_dir,
            taxonomy=taxonomy,
            max_file_bytes=int(self.max_file_bytes_var.get()),
            allowed_extensions=allowed_extensions,
            skip_dirs=skip_dirs,
            skip_hidden=self.skip_hidden_var.get(),
            skip_extensions=skip_extensions,
            use_exif=self.use_exif_var.get(),
            use_pdf_meta=self.use_pdf_meta_var.get(),
            use_ocr=self.use_ocr_var.get(),
            cluster_method=self.cluster_method_var.get().strip() or "centroid",
            cluster_max_size=int(self.cluster_max_size_var.get()),
            labeler_temperature=float(self.labeler_temp_var.get()),
            local_name_rules=self.cfg.local_name_rules,
            local_path_rules=self.cfg.local_path_rules,
            include_subfolders=self.include_subfolders_var.get(),
        )

    def _run(self, dry_run: bool) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        if not os.getenv("OPENAI_API_KEY"):
            messagebox.showerror("Missing API Key", "OPENAI_API_KEY is not set in the environment.")
            return

        cfg = self._gather_config()
        cfg.dry_run = dry_run
        save_config(cfg)

        self._reset_progress()
        self._set_buttons_state(tk.DISABLED)
        self.cancel_event.clear()
        self.log("Starting run...")

        def worker() -> None:
            try:
                organizer = Organizer(
                    cfg,
                    logger=self.log_queue.put,
                    progress=self.log_queue.put,
                    should_stop=self.cancel_event.is_set,
                )
                planned = organizer.plan()
                self.log_queue.put(f"Planned moves: {len(planned)}")
                if dry_run:
                    self.log_queue.put("Dry run complete. No files moved.")
                else:
                    organizer.apply(planned)
                    self.log_queue.put("Apply complete.")
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}")
            finally:
                self.root.after(0, lambda: self._set_buttons_state(tk.NORMAL))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _set_buttons_state(self, state: str) -> None:
        self.save_btn.config(state=state)
        self.dry_run_btn.config(state=state)
        self.apply_btn.config(state=state)
        if state == tk.NORMAL:
            self.cancel_btn.config(state=tk.DISABLED)
        else:
            self.cancel_btn.config(state=tk.NORMAL)

    def _pump_logs(self) -> None:
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if isinstance(msg, dict):
                self._handle_progress_event(msg)
            else:
                self.log(str(msg))
        self.root.after(200, self._pump_logs)

    def _reset_progress(self) -> None:
        self.scan_progress.stop()
        self.scan_status.config(text="Scanned: 0 files")
        self.ai_progress.config(maximum=0, value=0)
        self.ai_status.config(text="Embeddings: 0/0 | Labels: 0/0")
        self._embeddings_total = 0
        self._embeddings_done = 0
        self._labels_total = 0
        self._labels_done = 0

    def _handle_progress_event(self, event: dict) -> None:
        event_type = event.get("type")
        value = int(event.get("value", 0))

        if event_type == "scan_start":
            self.scan_progress.start(10)
        elif event_type == "scan_tick":
            self.scan_status.config(text=f"Scanned: {value} files")
        elif event_type == "scan_done":
            self.scan_progress.stop()
            self.scan_status.config(text=f"Scanned: {value} files")
        elif event_type == "embeddings_total":
            self._embeddings_total = value
            self._update_ai_status()
        elif event_type == "embeddings_done":
            self._embeddings_done = value
            self._update_ai_status()
        elif event_type == "label_total":
            self._labels_total = value
            self._update_ai_status()
        elif event_type == "label_done":
            self._labels_done = value
            self._update_ai_status()

    def _update_ai_status(self) -> None:
        total = self._embeddings_total + self._labels_total
        done = self._embeddings_done + self._labels_done
        self.ai_progress.config(maximum=max(total, 1), value=done)
        self.ai_status.config(
            text=(
                f"Embeddings: {self._embeddings_done}/{self._embeddings_total} | "
                f"Labels: {self._labels_done}/{self._labels_total}"
            )
        )

    def _toggle_advanced(self) -> None:
        if self.advanced_visible:
            self.advanced_frame.pack_forget()
            self.advanced_btn.config(text="Advanced ▸")
            self.advanced_visible = False
        else:
            self.advanced_frame.pack(fill=tk.X, pady=(0, 10))
            self.advanced_btn.config(text="Advanced ▾")
            self.advanced_visible = True

    def log(self, msg: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _cancel_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self.log("Cancel requested... stopping after current step.")


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
