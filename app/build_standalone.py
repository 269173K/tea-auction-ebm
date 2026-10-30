"""Build app/standalone.html: the same page with CSS, JS and the model bundle inlined.
It runs with no server (the browser twin of the EBM computes the forecasts), which is
handy for sharing a demo link. The Flask app remains the main front-end."""
from pathlib import Path

APP = Path(__file__).resolve().parent
html = (APP / "templates" / "index.html").read_text()
css = (APP / "static" / "style.css").read_text()
js = (APP / "static" / "app.js").read_text()
bundle = (APP / "static" / "model_bundle.json").read_text()

body = html[html.index("<!--PAGE-->"):html.index("<!--BUNDLE-->")]
head = """<title>Tea Auction Outlook</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
"""
out = (head + f"<style>\n{css}\nbody {{ background: var(--sage); color: var(--leaf); }}\n</style>\n" + body +
       f"<script>window.EMBEDDED_BUNDLE = {bundle};</script>\n<script>\n{js}\n</script>\n")
(APP / "standalone.html").write_text(out)
print("standalone.html", round(len(out) / 1024, 1), "KB")
