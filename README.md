# Smart Macbook Organise (Organized AI)

A desktop GUI tool for organizing files using smart rules plus OpenAI embeddings. It scans target folders, groups files, and proposes a clean folder structure. Designed to be safe by default: non‑recursive scanning, skip hidden files, and skip heavy extensions.

## Features
- GUI with progress bars and cancel button
- Non‑recursive by default (only top‑level files)
- Skip hidden files and heavy extensions (jar/pkg/dmg/zip/dng/etc.)
- Two‑stage AI labeling with a controlled taxonomy
- On‑disk embedding cache for lower cost and faster reruns
- Dry run mode before applying moves

## Requirements
- Python 3.10+
- OpenAI API key

Optional (only if you enable in Advanced):
- `Pillow` for EXIF metadata
- `pypdf` for PDF metadata

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional deps:
```bash
pip install Pillow pypdf
```

## Configure API Key

Set it in your shell before running:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

## Run the App

```bash
organized-ai
```

If you don’t want to install as a package, you can also run:

```bash
python -m organized_ai.app
```

## Usage Guide

1. **Select target folders** (e.g., Downloads, Documents).
2. **Run Dry** to preview the plan.
3. Check logs and progress bars.
4. Click **Apply** to move files.

### Important Defaults
- **Non‑recursive**: only top‑level files are processed. Subfolders are ignored unless you enable “Include subfolders (recursive)” in Advanced.
- **Skip hidden files/dirs**: enabled by default.
- **Skip extensions**: default list includes `jar`, `pkg`, `dmg`, `zip`, `dng`, `mp4`, etc.
- **Budget limit**: default `0.80` USD. Increase if you want more AI labeling.

### Advanced Options
Open **Advanced** to customize:
- Include subfolders (recursive)
- Cache directory
- Max file size for text reads
- Allowed extensions (allowlist)
- Skip extensions (denylist)
- Skip directories
- Taxonomy categories
- Clustering settings and labeler temperature

## Safety Notes
- Always use **Dry Run** first.
- Files already inside the destination folder are ignored.
- Moves are collision‑safe: existing destination files won’t be overwritten.

## Troubleshooting

**“Budget would exceed”**
- Lower scope (remove subfolders, skip more extensions), or increase budget.

**Slow or stuck run**
- Large folders can take time. Use skip directories and skip extensions.
- Check the progress bars for scan vs AI stages.

## Repository
GitHub: https://github.com/saaga112/Smart_Macbook_Organise
