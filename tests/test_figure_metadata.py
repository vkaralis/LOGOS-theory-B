"""Metadata removal must preserve the rendered image and SVG geometry."""
from PIL import Image, PngImagePlugin
from logos_b.figure_metadata import strip_figure_metadata


def test_png_metadata_removed_without_changing_pixels(tmp_path):
    path = tmp_path / "test.png"
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 128))
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "test generator")
    image.save(path, pnginfo=info, dpi=(160, 160))
    strip_figure_metadata(path)
    with Image.open(path) as clean:
        assert clean.info == {}
        assert clean.tobytes() == image.tobytes()


def test_svg_metadata_removed_without_changing_geometry(tmp_path):
    path = tmp_path / "test.svg"
    path.write_text('<svg><metadata><creator>test</creator></metadata><path d="M0 0L1 1"/></svg>')
    strip_figure_metadata(path)
    assert path.read_text() == '<svg><path d="M0 0L1 1"/></svg>'
