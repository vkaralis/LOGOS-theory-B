"""Remove optional plot metadata while preserving pixels and SVG geometry."""
import re
from pathlib import Path


def strip_figure_metadata(path):
    path = Path(path)
    if path.suffix == ".svg":
        path.write_text(re.sub(r"\s*<metadata>.*?</metadata>", "", path.read_text(), flags=re.DOTALL))
        return
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected a PNG")
    chunks = [data[:8]]
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        end = offset + length + 12
        if end > len(data):
            raise ValueError("Truncated PNG")
        kind = data[offset + 4:offset + 8]
        # Critical image chunks and transparency carry rendering data.
        if kind in {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}:
            chunks.append(data[offset:end])
        offset = end
    path.write_bytes(b"".join(chunks))
