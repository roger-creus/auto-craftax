#!/usr/bin/env python3
"""Convert report.md to a self-contained HTML file with embedded base64 images."""
import base64
import os
import re
import markdown

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(REPORT_DIR, "report.md")
HTML_PATH = os.path.join(REPORT_DIR, "report.html")
PLOTS_DIR = os.path.join(REPORT_DIR, "plots")

# Read markdown
with open(MD_PATH, "r") as f:
    md_content = f.read()

# Replace image references with base64-embedded images
def embed_image(match):
    alt_text = match.group(1)
    img_path = match.group(2)
    full_path = os.path.join(REPORT_DIR, img_path)
    if os.path.exists(full_path):
        with open(full_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lstrip(".")
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        return f'<img src="data:{mime};base64,{b64}" alt="{alt_text}" style="max-width: 100%; height: auto; margin: 1em 0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">'
    return match.group(0)

md_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed_image, md_content)

# Convert markdown to HTML
html_body = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc', 'nl2br'],
    extension_configs={
        'codehilite': {'css_class': 'highlight'},
    }
)

# Full HTML with CSS
html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Craftax-Symbolic-v1 Research Report</title>
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.7;
        color: #1a1a2e;
        background: #f0f2f5;
        padding: 2em 1em;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: white;
        padding: 3em;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
    }}
    h1 {{
        font-size: 2em;
        color: #1a1a2e;
        border-bottom: 3px solid #2196F3;
        padding-bottom: 0.5em;
        margin-bottom: 0.5em;
        margin-top: 1em;
    }}
    h1:first-child {{
        margin-top: 0;
    }}
    h2 {{
        font-size: 1.5em;
        color: #16213e;
        margin-top: 2em;
        margin-bottom: 0.5em;
        padding-bottom: 0.3em;
        border-bottom: 1px solid #e0e0e0;
    }}
    h3 {{
        font-size: 1.2em;
        color: #0f3460;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }}
    h4 {{
        font-size: 1.05em;
        color: #333;
        margin-top: 1.2em;
        margin-bottom: 0.3em;
    }}
    p {{
        margin-bottom: 1em;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1em 0;
        font-size: 0.9em;
        overflow-x: auto;
        display: block;
    }}
    th, td {{
        padding: 0.6em 0.8em;
        text-align: left;
        border: 1px solid #e0e0e0;
    }}
    th {{
        background: #f5f7fa;
        font-weight: 600;
        color: #1a1a2e;
        white-space: nowrap;
    }}
    tr:nth-child(even) {{
        background: #fafbfc;
    }}
    tr:hover {{
        background: #f0f4ff;
    }}
    code {{
        background: #f5f7fa;
        padding: 0.15em 0.4em;
        border-radius: 3px;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
        font-size: 0.88em;
        color: #d63384;
    }}
    pre {{
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 1.2em;
        border-radius: 8px;
        overflow-x: auto;
        margin: 1em 0;
        font-size: 0.85em;
        line-height: 1.5;
    }}
    pre code {{
        background: none;
        color: inherit;
        padding: 0;
    }}
    blockquote {{
        border-left: 4px solid #2196F3;
        padding: 0.5em 1em;
        margin: 1em 0;
        background: #f5f7fa;
        border-radius: 0 4px 4px 0;
    }}
    strong {{
        color: #1a1a2e;
    }}
    hr {{
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2em 0;
    }}
    ul, ol {{
        margin-left: 1.5em;
        margin-bottom: 1em;
    }}
    li {{
        margin-bottom: 0.3em;
    }}
    img {{
        display: block;
        margin: 1.5em auto;
    }}
    .status-badge {{
        display: inline-block;
        padding: 0.2em 0.6em;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }}
    @media (max-width: 768px) {{
        body {{
            padding: 0.5em;
        }}
        .container {{
            padding: 1.5em;
            border-radius: 0;
        }}
        h1 {{
            font-size: 1.5em;
        }}
        table {{
            font-size: 0.8em;
        }}
    }}
    @media print {{
        body {{
            background: white;
            padding: 0;
        }}
        .container {{
            box-shadow: none;
            padding: 1em;
        }}
    }}
</style>
</head>
<body>
<div class="container">
{html_body}
</div>
</body>
</html>"""

with open(HTML_PATH, "w") as f:
    f.write(html_full)

print(f"HTML report generated: {HTML_PATH}")
print(f"File size: {os.path.getsize(HTML_PATH) / 1024 / 1024:.1f} MB")
