#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-27 17:55:12
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/image_gallery_generator.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-27 17:55:12
# **********************************************************

"""
Generates a single self-contained HTML gallery page for eyeballing a
tree of generated images (posture variants, posture x emotion combos,
etc.) -- one <img> per file, shown large enough to judge quality, with
its bare filename (no path) underneath so you can locate it again with
`find <root> -name <filename>` and redo it if it's unacceptable.

No server, no third-party dependencies -- open the resulting
gallery.html directly in any browser (Linux or macOS). Images are
referenced by relative file:// path from the html, so it works
whether it's viewed right where it was generated or after rsync'ing
the whole assets/ tree elsewhere, as long as the relative layout is
preserved.

Images are grouped into a collapsible <details> section per parent
directory (a flat few-thousand-image page is unreadable otherwise). A
text box filters cards by filename substring, live. A "redo" checkbox
under each image lets you mark it; "Copy marked filenames" puts the
bare names of everything checked onto the clipboard, ready to paste
into a redo list.

Lives at <proj-root>/src/cloud_img_procurement/image_gallery_generator.py.
No editable install needed -- this script only touches the stdlib.

Usage:
    python src/cloud_img_procurement/image_gallery_generator.py \\
        --root assets/client_library_animated \\
        --out  assets/client_library_animated/gallery.html
"""

import argparse
import html
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

_PAGE_HEAD = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Image Gallery -- @@ROOT@@</title>
<style>
  :root { --card-width: 420px; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
         padding: 16px 24px 64px; background:#111; color:#eee; }
  header { position: sticky; top:0; background:#111; padding:12px 0;
           border-bottom:1px solid #333; z-index:10; }
  h1 { font-size: 16px; margin: 0 0 8px; color:#aaa; font-weight: normal; }
  .controls { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  input[type=text] { background:#222; border:1px solid #444; color:#eee;
                      padding:6px 10px; border-radius:4px; width:280px;
                      font-size:14px; }
  input[type=range] { width:160px; }
  button { background:#2a6; border:none; color:#fff; padding:6px 14px;
           border-radius:4px; cursor:pointer; font-size:14px; }
  button:hover { background:#3b7; }
  .stat { color:#888; font-size:13px; }
  section { margin-top: 22px; }
  h2 { font-size:14px; color:#9cf; border-bottom:1px solid #333;
       padding-bottom:4px; cursor:pointer; }
  .count { color:#666; font-weight:normal; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(var(--card-width), 1fr));
          gap:16px; margin-top:10px; }
  figure.card { margin:0; background:#1a1a1a; border:1px solid #333;
                border-radius:6px; padding:8px; }
  figure.card img { width:100%; height:auto; display:block; border-radius:4px;
                     background:#000; }
  figcaption { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               font-size:12px; color:#ddd; margin-top:6px; word-break: break-all;
               display:flex; justify-content:space-between; align-items:center;
               gap:6px; }
  .mark { color:#f88; font-family: sans-serif; font-size:12px; white-space:nowrap; }
  .hidden { display:none !important; }
</style>
</head>
<body>
<header>
  <h1>@@ROOT@@ &nbsp;&middot;&nbsp;
    <span class="stat">@@TOTAL@@ images in @@FOLDERS@@ folders</span></h1>
  <div class="controls">
    <input type="text" id="filter" placeholder="filter by filename...">
    <label class="stat">size <input type="range" id="size" min="200" max="800" value="420"></label>
    <button id="copyBtn">Copy marked filenames</button>
    <span class="stat">marked: <span id="markCount">0</span></span>
  </div>
</header>
"""

_PAGE_TAIL = """
<script>
const filterInput = document.getElementById('filter');
const sizeInput = document.getElementById('size');
const copyBtn = document.getElementById('copyBtn');
const markCount = document.getElementById('markCount');
const copyBtnOriginalText = copyBtn.textContent;

sizeInput.addEventListener('input', () => {
  document.documentElement.style.setProperty('--card-width', sizeInput.value + 'px');
});

function applyFilter() {
  const q = filterInput.value.trim().toLowerCase();
  document.querySelectorAll('section').forEach(sec => {
    let anyVisible = false;
    sec.querySelectorAll('figure.card').forEach(card => {
      const match = !q || card.dataset.filename.toLowerCase().includes(q);
      card.classList.toggle('hidden', !match);
      if (match) anyVisible = true;
    });
    sec.classList.toggle('hidden', !anyVisible);
    if (q && anyVisible) sec.querySelector('details') && (sec.querySelector('details').open = true);
  });
}
filterInput.addEventListener('input', applyFilter);

function updateMarkCount() {
  markCount.textContent = document.querySelectorAll('input.redo:checked').length;
}
document.querySelectorAll('input.redo').forEach(cb => cb.addEventListener('change', updateMarkCount));

copyBtn.addEventListener('click', () => {
  const names = Array.from(document.querySelectorAll('input.redo:checked'))
    .map(cb => cb.closest('figure').dataset.filename);
  const textarea = document.createElement('textarea');
  textarea.value = names.join('\\n');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  try { document.execCommand('copy'); } catch (e) { /* clipboard unavailable */ }
  document.body.removeChild(textarea);
  copyBtn.textContent = 'Copied ' + names.length;
  setTimeout(() => { copyBtn.textContent = copyBtnOriginalText; }, 1200);
});
</script>
</body>
</html>
"""


class ImageGalleryGenerator:
    """Builds a static HTML gallery for every image under a root dir.

    :param root: Directory to scan recursively for images.
    :type root: Path
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")

    #------------------------------------
    # _discover_images
    #-------------------

    def _discover_images(self) -> dict:
        """Finds every image under root, grouped by parent directory.

        :return: Dict mapping parent-dir Path -> sorted list of image
            Paths, itself sorted by parent-dir path.
        :rtype: dict
        """
        groups = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                groups.setdefault(path.parent, []).append(path)
        if not groups:
            raise FileNotFoundError(f"No images found under {self.root}")
        return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))

    #------------------------------------
    # _render_section
    #-------------------

    def _render_section(self, folder: Path, images: list, out_dir: Path) -> str:
        """Renders one collapsible <details> section for a folder's images.

        :param folder: The parent directory these images share.
        :type folder: Path
        :param images: Image paths in that folder.
        :type images: list
        :param out_dir: Directory the gallery HTML will be written into,
            used to compute relative image paths.
        :type out_dir: Path
        :return: HTML for this section.
        :rtype: str
        """
        rel_folder = folder.relative_to(self.root)
        cards = []
        for img in images:
            rel_img = Path(os.path.relpath(img, start=out_dir)).as_posix()
            name = html.escape(img.name)
            cards.append(
                f'<figure class="card" data-filename="{name}">'
                f'<a href="{rel_img}" target="_blank">'
                f'<img src="{rel_img}" loading="lazy" alt="{name}"></a>'
                f'<figcaption>{name}'
                f'<label class="mark"><input type="checkbox" class="redo"> redo</label>'
                f'</figcaption>'
                f'</figure>'
            )
        return (
            f'<section><details open><summary><h2 style="display:inline">'
            f'{html.escape(rel_folder.as_posix())} '
            f'<span class="count">({len(images)})</span></h2></summary>'
            f'<div class="grid">{"".join(cards)}</div></details></section>'
        )

    #------------------------------------
    # generate
    #-------------------

    def generate(self, out_path: Path) -> Path:
        """Builds the gallery HTML and writes it to out_path.

        :param out_path: Where to write the gallery HTML file. Its
            parent directory is what all image paths are computed
            relative to.
        :type out_path: Path
        :return: The path written.
        :rtype: Path
        """
        groups = self._discover_images()
        total = sum(len(v) for v in groups.values())
        log.info("Found %d images in %d folders under %s", total, len(groups), self.root)

        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sections = "\n".join(
            self._render_section(folder, images, out_path.parent)
            for folder, images in groups.items()
        )

        head = (
            _PAGE_HEAD
            .replace("@@ROOT@@", html.escape(str(self.root)))
            .replace("@@TOTAL@@", str(total))
            .replace("@@FOLDERS@@", str(len(groups)))
        )
        out_path.write_text(head + sections + _PAGE_TAIL, encoding="utf-8")
        log.info("Wrote gallery: %s", out_path)
        log.info("Open it in a browser: file://%s", out_path)
        return out_path


class ImageGalleryGeneratorCLI:
    """Parses CLI arguments and runs ImageGalleryGenerator.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True,
                             help="Directory to scan recursively for images "
                                  "(e.g. assets/client_library_animated).")
        parser.add_argument("--out", type=Path, default=None,
                             help="Output gallery HTML path (default: "
                                  "<root>/gallery.html).")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Builds the gallery and reports where it landed."""
        generator = ImageGalleryGenerator(self.args.root)
        out_path = self.args.out or (Path(self.args.root) / "gallery.html")
        generator.generate(out_path)


if __name__ == "__main__":
    ImageGalleryGeneratorCLI().run()
