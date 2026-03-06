import argparse
import html
import json
import os
import re
from itertools import zip_longest


def safe_sentence_tokenize(text: str) -> list:
    """
    Tokenize text into sentences.
    Tries NLTK if available; falls back to a simple regex-based splitter.
    """
    if not text:
        return []
    try:
        import nltk

        # Some environments may not have punkt installed; try to download silently
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
        if sentences:
            return sentences
    except Exception:
        pass

    # Fallback: split on common sentence terminators followed by whitespace/newline
    # Keep the delimiter by using a capturing group, then recombine for readability
    parts = re.split(r"([.!?]+)\s+", text)
    merged = []
    for i in range(0, len(parts), 2):
        base = parts[i] if i < len(parts) else ""
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (base + punct).strip()
        if chunk:
            merged.append(chunk)
    # If nothing matched (e.g., no punctuation), treat as single sentence
    return merged if merged else [text.strip()]


def generate_html(pairs: list, title: str) -> str:
    """
    pairs: list of dicts with keys: index, label, predict
    """
    def render_sample(idx: int, label: str, predict: str, prompt: str) -> str:
        label_sents = safe_sentence_tokenize(label)
        pred_sents = safe_sentence_tokenize(predict)

        rows = []
        for row_idx, (ls, ps) in enumerate(zip_longest(label_sents, pred_sents, fillvalue="")):
            rows.append(
                f"""
                <tr>
                    <td class=\"row-idx\">{row_idx + 1}</td>
                    <td class=\"label\">{html.escape(ls)}</td>
                    <td class=\"predict\">{html.escape(ps)}</td>
                </tr>
                """
            )

        preview_left = html.escape((label or "").strip().replace("\n", " ")[:140])
        preview_right = html.escape((predict or "").strip().replace("\n", " ")[:140])
        preview_prompt = html.escape((prompt or "").strip().replace("\n", " ")[:140])

        return f"""
        <details>
          <summary>
            <span class=\"sample-id\">#{idx}</span>
            <span class=\"preview prompt\">Prompt: {preview_prompt}</span>
            <span class=\"preview left\">Label: {preview_left}</span>
            <span class=\"preview right\">Pred: {preview_right}</span>
          </summary>
          <div class=\"prompt-block\">
            <div class=\"prompt-title\">Prompt</div>
            <pre class=\"prompt-text\">{html.escape(prompt or "")}</pre>
          </div>
          <table class=\"pair-table\" aria-label=\"label-prediction sentence alignment\">
            <thead>
              <tr>
                <th class=\"row-idx\">#</th>
                <th>Label</th>
                <th>Prediction</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </details>
        """

    items_html = [render_sample(p["index"], p["label"], p["predict"], p.get("prompt", "")) for p in pairs]

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --subtle: #9ca3af;
      --border: #374151;
      --accent: #22d3ee;
      --rowOdd: rgba(255, 255, 255, 0.02);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\";
      background: var(--bg); color: var(--text);
    }}
    header {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
    .meta {{ color: var(--subtle); font-size: 13px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin: 8px 0 16px; }}
    input[type='text'] {{
      background: var(--panel); border: 1px solid var(--border); color: var(--text);
      padding: 6px 10px; border-radius: 6px; width: 320px;
    }}
    button {{ background: var(--panel); color: var(--text); border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; cursor: pointer; }}
    button:hover {{ border-color: var(--accent); }}

    details {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 10px 0;
      padding: 8px 10px;
    }}
    summary {{ cursor: pointer; display: flex; gap: 12px; align-items: center; color: var(--subtle); }}
    .sample-id {{ color: var(--accent); font-weight: 600; }}
    .preview {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 40ch; }}
    .preview.right {{ color: #60a5fa; }}
    .preview.left {{ color: #f472b6; }}
    .preview.prompt {{ color: #34d399; }}

    table.pair-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; table-layout: fixed; }}
    thead th {{
      position: sticky; top: 0; background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(6px);
      z-index: 1; text-align: left; font-weight: 600; padding: 8px; border-bottom: 1px solid var(--border);
    }}
    tbody td {{ vertical-align: top; padding: 8px; border-bottom: 1px solid var(--border); }}
    tbody tr:nth-child(odd) {{ background: var(--rowOdd); }}
    td.row-idx, th.row-idx {{ width: 56px; color: var(--subtle); }}
    td.label {{ color: #f9a8d4; white-space: pre-wrap; }}
    td.predict {{ color: #93c5fd; white-space: pre-wrap; }}

    .prompt-block {{
      margin-top: 10px; border: 1px solid var(--border); border-radius: 6px;
      background: rgba(255,255,255,0.02);
    }}
    .prompt-title {{ font-size: 12px; color: var(--subtle); padding: 6px 8px; border-bottom: 1px solid var(--border); }}
    .prompt-text {{ margin: 0; padding: 10px 12px; white-space: pre-wrap; color: #d1fae5; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class=\"meta\">Sentence-aligned side-by-side comparison</div>
  </header>

  <div class=\"controls\">
    <input id=\"filter\" type=\"text\" placeholder=\"Filter by substring...\" />
    <button id=\"expandAll\">Expand all</button>
    <button id=\"collapseAll\">Collapse all</button>
  </div>

  <div id=\"content\">{''.join(items_html)}</div>

  <script>
    const filter = document.getElementById('filter');
    const content = document.getElementById('content');
    const expandAll = document.getElementById('expandAll');
    const collapseAll = document.getElementById('collapseAll');

    function applyFilter() {{
      const q = filter.value.toLowerCase();
      for (const details of content.querySelectorAll('details')) {{
        const text = details.innerText.toLowerCase();
        details.style.display = q && !text.includes(q) ? 'none' : '';
      }}
    }}
    filter.addEventListener('input', applyFilter);
    expandAll.addEventListener('click', () => {{ document.querySelectorAll('details').forEach(d => d.open = true); }});
    collapseAll.addEventListener('click', () => {{ document.querySelectorAll('details').forEach(d => d.open = false); }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML side-by-side comparison of labels and predictions from a JSONL file.")
    parser.add_argument("--jsonl", type=str, default="output/step1_inversion/eval/Qwen2.5-7B-Instruct-full-sft-inversion/generated_predictions.jsonl", help="Path to generated_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of pairs to include")
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = obj.get("label", "")
            predict = obj.get("predict", "")
            prompt = obj.get("prompt", "")
            pairs.append({"index": idx, "label": label, "predict": predict, "prompt": prompt})
            if args.limit is not None and len(pairs) >= args.limit:
                break

    title = f"Comparison: {os.path.basename(os.path.dirname(jsonl_path))} ({len(pairs)} pairs)"
    html_str = generate_html(pairs, title)

    out_path = os.path.join(os.path.dirname(jsonl_path), "comparison.html")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(html_str)

    print(f"Wrote HTML comparison to: {out_path}")


if __name__ == "__main__":
    main()


