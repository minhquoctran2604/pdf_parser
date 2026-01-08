from pathlib import Path
from pdf_parser import PDFParser

PDF_PATH = "path/to/file.pdf"
OUT_PATH = "test_output.md"

parser = PDFParser()
doc = parser.parse(PDF_PATH)
if doc is None:
    raise SystemExit("Parse failed")

Path(OUT_PATH).write_text(doc.enriched_markdown, encoding="utf-8")
print(f"Saved: {OUT_PATH}")
