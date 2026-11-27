"""Build site/index.html (the standalone page GitHub Pages serves) from site/page.html.

    python scripts/build_site.py

site/page.html is the source: the head block between the <!--head--> markers, then the body.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPEN, CLOSE = "<!--head-->", "<!--/head-->"
_SVG = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' "
        "fill='%2315203a'/%3E%3Cpath d='M5 9h8v7h8v7h6' fill='none' stroke='%2386a9ff' stroke-width='3'/%3E%3C/svg%3E")
ICON = f'<link rel="icon" href="data:image/svg+xml,{_SVG}">'


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
<meta property="og:image" content="https://raw.githubusercontent.com/mvplab-ai/agentdream/main/docs/social-preview.png">
<meta name="twitter:card" content="summary_large_image">
{ICON}
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
