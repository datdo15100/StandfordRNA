#!/usr/bin/env python
"""Render the two Vietnamese thesis guides as printable A4 PDFs."""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import re
import subprocess

import markdown
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "pdf"
CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

DOCUMENTS = [
    {
        "source": ROOT / "reports/thesis_notes/end_to_end_flow_explained.md",
        "output": OUTPUT_DIR / "01_RNA3D_End_to_End_Flow_VI.pdf",
        "title": "RNA 3D: Từ sequence đến prediction",
        "subtitle": "Bản giải thích workflow ngắn, có ví dụ xuyên suốt",
    },
    {
        "source": ROOT / "reports/thesis_notes/thesis_defense_explanation_vi.md",
        "output": OUTPUT_DIR / "02_RNA3D_Thesis_Defense_Guide_VI.pdf",
        "title": "RNA 3D Thesis Defense Guide",
        "subtitle": "Method, workflow, ablation, results và câu hỏi phản biện",
    },
]

CSS = r"""
@page {
  size: A4;
  margin: 17mm 15mm 18mm 15mm;
  @bottom-center {
    content: counter(page);
    color: #64748b;
    font-size: 8.5pt;
  }
}
* { box-sizing: border-box; }
html {
  color: #172033;
  font-family: "DejaVu Sans", "Arial", sans-serif;
  font-size: 10.2pt;
  line-height: 1.58;
}
body { margin: 0; }
.cover {
  min-height: 245mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
  break-after: page;
}
.cover .eyebrow {
  color: #2563eb;
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}
.cover h1 {
  border: 0;
  color: #102a56;
  font-size: 27pt;
  line-height: 1.18;
  margin: 12mm auto 5mm;
  max-width: 165mm;
  padding: 0;
}
.cover .subtitle {
  color: #475569;
  font-size: 13pt;
  margin: 0 auto;
  max-width: 155mm;
}
.cover .meta {
  color: #64748b;
  font-size: 9pt;
  margin-top: 18mm;
}
.toc-wrapper {
  break-after: page;
  columns: 2;
  column-gap: 10mm;
  font-size: 8.8pt;
}
.toc-wrapper::before {
  color: #102a56;
  content: "Mục lục";
  display: block;
  font-size: 20pt;
  font-weight: 700;
  margin-bottom: 8mm;
  column-span: all;
}
.toc-wrapper ul { list-style: none; margin: 0; padding-left: 0; }
.toc-wrapper ul ul { padding-left: 4mm; }
.toc-wrapper li { break-inside: avoid; margin: 0 0 1.5mm; }
.toc-wrapper a { color: #334155; text-decoration: none; }
h1, h2, h3, h4 {
  color: #102a56;
  line-height: 1.25;
  page-break-after: avoid;
}
main > h1 {
  border-bottom: 1.5px solid #93c5fd;
  break-before: page;
  font-size: 20pt;
  margin: 0 0 7mm;
  padding-bottom: 3mm;
}
main > h1:first-child { break-before: auto; }
h2 { font-size: 15pt; margin: 8mm 0 3mm; }
h3 { font-size: 12pt; margin: 6mm 0 2mm; }
h4 { font-size: 10.5pt; margin: 5mm 0 2mm; }
p { margin: 0 0 3.2mm; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 4mm; padding-left: 7mm; }
li { margin-bottom: 1.2mm; }
blockquote {
  background: #eff6ff;
  border-left: 4px solid #3b82f6;
  color: #1e3a5f;
  margin: 5mm 0;
  padding: 4mm 5mm;
  break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
code {
  background: #eef2f7;
  border-radius: 3px;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.88em;
  padding: 0.2mm 1mm;
}
pre {
  background: #0f172a;
  border-radius: 5px;
  color: #e2e8f0;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.25pt;
  line-height: 1.42;
  margin: 4mm 0 5mm;
  overflow-wrap: anywhere;
  padding: 4mm;
  white-space: pre-wrap;
  break-inside: avoid;
}
pre code { background: transparent; color: inherit; padding: 0; }
table {
  border-collapse: collapse;
  font-size: 8.25pt;
  margin: 4mm 0 6mm;
  width: 100%;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  background: #dbeafe;
  color: #17345f;
  font-weight: 700;
  text-align: left;
}
th, td {
  border: 0.6px solid #b8c6d9;
  padding: 1.8mm 2mm;
  vertical-align: top;
  overflow-wrap: anywhere;
}
tbody tr:nth-child(even) { background: #f8fafc; }
hr {
  border: 0;
  border-top: 1px solid #cbd5e1;
  margin: 8mm 0;
}
a { color: #1d4ed8; }
strong { color: #142e55; }
"""


def windows_path(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def windows_file_url(path: Path) -> str:
    value = windows_path(path).replace("\\", "/")
    return f"file:///{value}"


def render_html(document: dict) -> Path:
    source = Path(document["source"])
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"\A\s*#\s+[^\n]+\n+", "", text, count=1)
    converter = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        extension_configs={"toc": {"permalink": False, "toc_depth": "1-3"}},
        output_format="html5",
    )
    body = converter.convert(text)
    html_path = Path(document["output"]).with_suffix(".print.html")
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="vi">',
                "<head>",
                '<meta charset="utf-8">',
                f"<title>{escape(document['title'])}</title>",
                f"<style>{CSS}</style>",
                "</head>",
                "<body>",
                '<section class="cover">',
                '<div class="eyebrow">Stanford RNA 3D Folding Thesis</div>',
                f"<h1>{escape(document['title'])}</h1>",
                f'<p class="subtitle">{escape(document["subtitle"])}</p>',
                f'<p class="meta">Generated {date.today().isoformat()}</p>',
                "</section>",
                f'<nav class="toc-wrapper">{converter.toc}</nav>',
                f"<main>{body}</main>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return html_path


def render_pdf(document: dict) -> tuple[int, int]:
    if not CHROME.exists():
        raise FileNotFoundError(f"Chrome not found: {CHROME}")
    output = Path(document["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    html_path = render_html(document)
    try:
        completed = subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={windows_path(output)}",
                windows_file_url(html_path),
            ],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Chrome PDF export failed for {output.name}: {completed.stderr}"
            )
        if not output.exists() or output.stat().st_size < 10_000:
            raise RuntimeError(f"missing or implausibly small PDF: {output}")
        reader = PdfReader(output)
        sample = "\n".join(
            (page.extract_text() or "") for page in reader.pages[: min(4, len(reader.pages))]
        )
        if "RNA" not in sample or len(sample) < 500:
            raise RuntimeError(f"PDF text validation failed: {output}")
        return len(reader.pages), output.stat().st_size
    finally:
        html_path.unlink(missing_ok=True)


def main() -> None:
    for document in DOCUMENTS:
        pages, size = render_pdf(document)
        relative = Path(document["output"]).relative_to(ROOT)
        print(f"{relative}: {pages} pages, {size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
