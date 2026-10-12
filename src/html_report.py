"""Turn the markdown reports into self-contained HTML (charts embedded as base64 data URIs).

The HTML files can be emailed or opened offline: no external CSS, fonts, scripts or image files are needed.
"""
from __future__ import annotations

import base64
import html
import mimetypes
import re
from pathlib import Path

import markdown

CSS = """
:root{--bg:#fbfbfa;--fg:#1d2330;--muted:#5d6676;--line:#dfe2e7;--head:#eef1f5;--accent:#1f4e79;--warn-bg:#fff4e0;--warn-line:#d98c1f}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313c;--head:#1d222b;--accent:#7fb2e5;--warn-bg:#2a2214;--warn-line:#c98a2b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 Georgia,"Times New Roman",serif}
main{max-width:880px;margin:0 auto;padding:40px 20px 72px}
h1,h2,h3,table,code,.meta{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:1.75rem;line-height:1.25;margin:0 0 8px}
h2{font-size:1.2rem;margin:38px 0 12px;padding-top:16px;border-top:1px solid var(--line);color:var(--accent)}
h3{font-size:1.02rem;margin:24px 0 8px}
p{margin:0 0 14px;max-width:74ch}ul,ol{margin:0 0 14px;padding-left:1.3rem;max-width:74ch}li{margin-bottom:4px}
code{font-size:.9em;background:var(--head);padding:1px 5px;border-radius:3px}
pre{background:var(--head);padding:12px;overflow-x:auto;border-radius:4px}pre code{background:none;padding:0}
blockquote{background:var(--warn-bg);border-left:4px solid var(--warn-line);margin:0 0 16px;padding:10px 16px;border-radius:0 4px 4px 0}
blockquote p{margin:0}
.scroll{overflow-x:auto;margin:0 0 14px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--head);border-bottom:2px solid var(--line);font-weight:600}
tbody tr:hover{background:var(--head)}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff;display:block;margin:14px 0}
em{color:var(--muted)}
.meta{color:var(--muted);font-size:.85rem;margin:0 0 24px}
"""

_IMG = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', flags=re.I)


def _embed_images(body: str, base_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        f = (base_dir / html.unescape(src)).resolve()
        if not f.is_file():
            return m.group(0)               # leave the relative link; the file is just missing
        mime = mimetypes.guess_type(f.name)[0] or "image/png"
        return m.group(1) + f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode("ascii") + m.group(3)
    return _IMG.sub(repl, body)


def markdown_to_html(md_text: str, base_dir: Path, title: str | None = None, subtitle: str | None = None) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists", "fenced_code"])
    body = body.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")
    body = _embed_images(body, Path(base_dir))
    if title is None:
        m = re.search(r"<h1>(.*?)</h1>", body, flags=re.S)
        title = re.sub(r"<[^>]+>", "", m.group(1)) if m else "Report"
    sub = f'<p class="meta">{html.escape(subtitle)}</p>' if subtitle else ""
    if sub and "</h1>" in body:
        body = body.replace("</h1>", "</h1>" + sub, 1)
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n')


def write_html(md_path: Path, html_path: Path | None = None, subtitle: str | None = None) -> Path:
    md_path = Path(md_path)
    html_path = Path(html_path or md_path.with_suffix(".html"))
    text = md_path.read_text(encoding="utf-8")
    html_path.write_text(markdown_to_html(text, md_path.parent, subtitle=subtitle), encoding="utf-8")
    return html_path


def write_html_reports(reports_dir: Path, names=("macro_outlook", "evaluation_report")) -> list[Path]:
    reports_dir = Path(reports_dir)
    return [write_html(reports_dir / f"{n}.md") for n in names if (reports_dir / f"{n}.md").exists()]
