"""Build Markdown, notebook and a watermark-free PDF from computed results.

python build_submission.py [--execute-notebook]
PDF rendering uses local math SVGs and headless Chromium; no CDN is needed.
"""

import argparse
import base64
import html
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
import markdown
import nbformat
from playwright.sync_api import sync_playwright

from submission_report import build_submission_report
from artifact_checks import verify_artifacts

ROOT = Path(__file__).resolve().parent
PDF_NAME = "Albert_Kabore_Project1_Report.pdf"
NOTEBOOK_NAME = "Project1_Sentiment_LR_FFNN.ipynb"


def make_notebook(execute=False):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("# Project 1: Logistic Regression and Feed-Forward Neural Network\n\n"
                "**Albert Kabore**\n\n"
                "This notebook uses the same implementation as `python train.py`. "
                "It loads and verifies saved results by default. Set `RECOMPUTE = True` "
                "below to retrain both models from the real CSV (a few minutes). "
                "Each report section is rendered from computed results, so there are no "
                "manually entered metric tables."),
             code("import json\nfrom pathlib import Path\nfrom IPython.display import HTML, Markdown, display\n"
                  "from train import run_experiment\n"
                  "from submission_report import build_submission_report\n"
                  "from artifact_checks import verify_artifacts\n\n"
                  "display(HTML('''<style>\n"
                  ".jp-RenderedHTMLCommon, .jp-RenderedHTMLCommon *,\n"
                  ".jp-RenderedMarkdown, .jp-RenderedMarkdown *,\n"
                  ".jp-OutputArea-output, .jp-OutputArea-output *,\n"
                  ".rendered_html, .rendered_html *, .text_cell_render, .text_cell_render *,\n"
                  ".output_area, .output_area *, .jp-CodeCell .cm-content,\n"
                  ".jp-CodeCell .cm-content *, .CodeMirror-code, .CodeMirror-code *\n"
                  "{ color: #000000 !important; }\n"
                  ".jp-RenderedHTMLCommon, .jp-RenderedMarkdown, .jp-OutputArea-output,\n"
                  ".rendered_html, .text_cell_render, .output_area\n"
                  "{ background-color: #ffffff; }\n"
                  "</style>'''))\n\n"
                  "ROOT = Path.cwd()\nRECOMPUTE = False\n"
                  "result_path = ROOT / 'results/results.json'\n"
                  "if RECOMPUTE or not result_path.exists():\n"
                  "    result = run_experiment()\n"
                  "else:\n"
                  "    result = json.loads(result_path.read_text(encoding='utf-8'))\n"
                  "verify_artifacts(result)\n"
                  "report = build_submission_report(result, ROOT / 'REPORT.md')\n"
                  "sections = report.split('\\n## ')\n"
                  "sections = [sections[0]] + ['## ' + part for part in sections[1:]]\n"
                  "print('Verified dataset, source code, split membership and prediction-file hashes.')")]
    section_count = (ROOT / "REPORT.md").read_text(encoding="utf-8").count("\n## ")
    for section in range(1, section_count + 1):
        cells.append(code(f"display(Markdown(sections[{section}]))"))
        if section == 7:
            cells.append(code("import inspect\nfrom sentiment import LogisticRegression, FeedForwardNetwork\n"
                              "print(inspect.getsource(LogisticRegression.gradient))\n"
                              "print(inspect.getsource(LogisticRegression.fit))\n"
                              "print(inspect.getsource(FeedForwardNetwork.gradients))"))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    if execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, ROOT / NOTEBOOK_NAME)


def math_image(expression, block):
    stream = io.BytesIO()
    math_to_image(f"${expression.strip()}$", stream,
                  prop=FontProperties(size=12 if block else 10.5), format="svg", color="#000000")
    svg = stream.getvalue()
    width = float(re.search(rb'width="([\d.]+)pt"', svg).group(1))
    encoded = base64.b64encode(svg).decode("ascii")
    tag = (f'<img class="math" style="width:{width:.2f}pt" '
           f'alt="{html.escape(expression, quote=True)}" src="data:image/svg+xml;base64,{encoded}">')
    return f'<div class="equation">{tag}</div>' if block else tag


def make_pdf(report):
    # Protect math before Markdown processes underscores and backslashes.
    math_tokens = []

    def save_math(match):
        expression = match.group(1)
        token = f"MATHEQUATIONTOKEN{len(math_tokens)}END"
        math_tokens.append(math_image(expression, block=True))
        return "\n\n" + token + "\n\n"

    prepared = re.sub(r"\$\$(.*?)\$\$", save_math, report, flags=re.DOTALL)
    body = markdown.markdown(prepared, extensions=["tables", "fenced_code"])
    for index, value in enumerate(math_tokens):
        token = f"MATHEQUATIONTOKEN{index}END"
        body = body.replace(f"<p>{token}</p>", value)
    css = """
        @page { size: Letter; margin: 0.7in 0.68in; }
        body { font: 10.5pt/1.42 'Segoe UI', Arial, sans-serif; color: #000000; }
        body, body * { color: #000000 !important; }
        h1 { font-size: 22pt; line-height: 1.15; margin: 0 0 16pt; }
        h2 { font-size: 15pt; margin: 20pt 0 8pt; break-after: avoid; }
        h3 { font-size: 12pt; margin: 14pt 0 6pt; break-after: avoid; }
        p { margin: 7pt 0; }
        a, a:visited { color: #000000; text-decoration: none; overflow-wrap: anywhere; }
        table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9pt; }
        thead { display: table-header-group; }
        th { background: #edf2f6; text-align: left; }
        td, th { padding: 4pt 5pt; border: 0.5pt solid #c9cfd5; overflow-wrap: anywhere; }
        tr { break-inside: avoid; }
        blockquote { border-left: 2pt solid #8da6bd; margin: 10pt 0; padding: 1pt 10pt; font-style: italic; }
        pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 8.3pt; line-height: 1.35;
              background: #f6f6f5; padding: 9pt; }
        code { font: 8.5pt Consolas, monospace; overflow-wrap: anywhere; }
        img { max-width: 100%; height: auto; }
        .equation { margin: 12pt 0; text-align: center; break-inside: avoid; }
        .math { max-width: 100%; vertical-align: middle; }
    """
    document = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Project 1 - Albert Kabore</title>"
                f"<base href='{ROOT.as_uri()}/'><style>{css}</style></head><body>{body}</body></html>")
    html_path = ROOT / "results/report.html"
    html_path.write_text(document, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        missing = page.locator("img").evaluate_all("imgs => imgs.filter(i => !i.complete || !i.naturalWidth).map(i => i.alt)")
        if missing:
            raise ValueError(f"Missing PDF images: {missing}")
        page.pdf(path=str(ROOT / PDF_NAME), format="Letter", print_background=True,
                 display_header_footer=False, prefer_css_page_size=True)
        browser.close()
    print(f"Built {PDF_NAME} without overlays or browser headers.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebook", action="store_true")
    args = parser.parse_args()
    result = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))
    verify_artifacts(result)
    report = build_submission_report(result, ROOT / "REPORT.md")
    make_notebook(execute=args.execute_notebook)
    make_pdf(report)


if __name__ == "__main__":
    main()
