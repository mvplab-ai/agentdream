"""Build site/index.html (the standalone page GitHub Pages serves) from site/page.html.

    python scripts/build_site.py

site/page.html is the source: the head block between the <!--head--> markers, then the body.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPEN, CLOSE = "<!--head-->", "<!--/head-->"


def build() -> str:
    src = (ROOT / "site" / "page.html").read_text(encoding="utf-8")
    head = src[src.index(OPEN) + len(OPEN):src.index(CLOSE)].strip()
    body = src[src.index(CLOSE) + len(CLOSE):].strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta property="og:title" content="agentdream">
<meta property="og:description" content="Your agent's notes, checked against what actually happened. \
Local-first memory consolidation with cited corrections.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://agentdream.dev/">
<link rel="canonical" href="https://agentdream.dev/">
{head}
</head>
<body style="margin:0">
{body}
</body>
</html>
"""


if __name__ == "__main__":
    (ROOT / "site" / "index.html").write_text(build(), encoding="utf-8")
    print("wrote site/index.html")
