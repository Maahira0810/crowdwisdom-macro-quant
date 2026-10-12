import base64

from src.html_report import markdown_to_html, write_html

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")


def test_tables_banner_and_unicode_survive(tmp_path):
    md = "# Title \u2014 test\n\n> **PROXY** warning\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = markdown_to_html(md, tmp_path)
    assert "<table>" in out and 'class="scroll"' in out
    assert "<blockquote>" in out and "\u2014" in out        # em dash kept (no mojibake)
    assert "<title>Title \u2014 test</title>" in out


def test_local_images_are_embedded_as_data_uri(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "x.png").write_bytes(PNG)
    out = markdown_to_html("![chart](figures/x.png)", tmp_path)
    assert "data:image/png;base64," in out and "figures/x.png" not in out


def test_missing_image_keeps_link_and_remote_untouched(tmp_path):
    out = markdown_to_html("![a](figures/missing.png)\n\n![b](https://example.com/b.png)", tmp_path)
    assert 'src="figures/missing.png"' in out and 'src="https://example.com/b.png"' in out


def test_write_html_roundtrip_utf8(tmp_path):
    md = tmp_path / "r.md"
    md.write_text("# R\n\nCaf\u00e9 \u2014 ok", encoding="utf-8")
    html_path = write_html(md)
    assert html_path.suffix == ".html" and "Caf\u00e9" in html_path.read_text(encoding="utf-8")
