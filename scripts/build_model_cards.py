"""Render per-model card RST pages from probe JSON under ``docs/_data/cards/``.

Inputs:  ``docs/_data/cards/<provider>/<model>.json`` (produced by
         ``python -m src.main -l`` per checkpoint).
Outputs: ``docs/models/<provider>/<model>.rst`` + ``docs/models/index.rst``.

Each card page embeds two raw-HTML views of the same data: a collapsible tree
(nested ``<details>``) for architectural browsing, and a flat sortable table
for grep / copy-paste into configs. No Sphinx extensions or JS toolchain are
required — just native browser features plus a tiny inline sort helper.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / 'docs' / '_data' / 'cards'
OUT_DIR = ROOT / 'docs' / 'models'

SORT_JS = """
<script>
(function(){
  function parseNum(s){var n = Number(String(s).replace(/[,_]/g,''));return isNaN(n)?null:n;}
  document.querySelectorAll('table.vlm-sortable').forEach(function(tbl){
    tbl.querySelectorAll('th').forEach(function(th, idx){
      th.style.cursor = 'pointer';
      var asc = true;
      th.addEventListener('click', function(){
        var tbody = tbl.tBodies[0];
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function(a,b){
          var x = a.cells[idx].innerText.trim();
          var y = b.cells[idx].innerText.trim();
          var xn = parseNum(x), yn = parseNum(y);
          if(xn !== null && yn !== null){return asc?xn-yn:yn-xn;}
          return asc ? x.localeCompare(y) : y.localeCompare(x);
        });
        asc = !asc;
        rows.forEach(function(r){tbody.appendChild(r);});
      });
    });
  });
})();
</script>
"""


def _human_params(n: int) -> str:
    if n == 0:
        return '0'
    for unit, scale in (('B', 1e9), ('M', 1e6), ('K', 1e3)):
        if n >= scale:
            return f'{n / scale:.2f}{unit}'
    return str(n)


def _shape_str(shape_info: Any) -> str:
    """Render the output_shapes field of a module record into a short label.

    Args:
        shape_info: Value from a card's ``output_shapes`` field (dict, list, or
            scalar-like) as produced by ``src.model_card._summarize``.

    Returns:
        A compact string representation suitable for HTML/RST rendering.
    """
    if shape_info is None:
        return '—'
    if isinstance(shape_info, dict):
        if 'shape' in shape_info and 'dtype' in shape_info:
            shp = shape_info['shape']
            return f'({", ".join(str(x) for x in shp)}) {shape_info["dtype"]}'
        if 'type' in shape_info and set(shape_info.keys()) == {'type'}:
            return shape_info['type']
        # ModelOutput-like
        parts = [f'{k}={_shape_str(v)}' for k, v in shape_info.items()]
        return '{' + ', '.join(parts) + '}'
    if isinstance(shape_info, list):
        if not shape_info:
            return '—'
        return '[' + ', '.join(_shape_str(x) for x in shape_info) + ']'
    return str(shape_info)


def _build_tree(modules: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a nested dict tree from dotted module names.

    Args:
        modules: Flat list of module records from a card JSON.

    Returns:
        Nested tree where each node has ``_self`` (the record, if any) and
        ``_children`` (dict of child-name to sub-node).
    """
    root: dict[str, Any] = {'_self': None, '_children': {}}
    for m in modules:
        parts = m['name'].split('.') if m['name'] else []
        node = root
        for p in parts:
            node = node['_children'].setdefault(p, {'_self': None, '_children': {}})
        node['_self'] = m
    return root


def _render_tree_html(node: dict[str, Any], label: str = '<root>') -> list[str]:
    self_rec = node.get('_self')
    children = node.get('_children', {})
    lines: list[str] = []
    if self_rec is not None:
        cls = self_rec['class']
        pd = self_rec['params_direct']
        pt = self_rec['params_total']
        out = _shape_str(self_rec['output_shapes'])
        summary_bits = [f'<code>{html.escape(label)}</code>',
                        f'<em>{html.escape(cls)}</em>']
        if pt > 0:
            summary_bits.append(f'params={_human_params(pt)}')
            if pd not in (0, pt):
                summary_bits.append(f'direct={_human_params(pd)}')
        summary_bits.append(f'out={html.escape(out)}')
        summary = ' · '.join(summary_bits)
    else:
        summary = f'<code>{html.escape(label)}</code>'

    if children:
        lines.append(f'<details open><summary>{summary}</summary>')
        lines.append('<div style="margin-left:1.2em">')
        for name, child in children.items():
            lines.extend(_render_tree_html(child, name))
        lines.append('</div></details>')
    else:
        lines.append(f'<div>{summary}</div>')
    return lines


def _render_table_html(modules: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for m in modules:
        name = html.escape(m['name']) or '<em>&lt;root&gt;</em>'
        cls = html.escape(m['class'])
        p = m['params_total']
        out = html.escape(_shape_str(m['output_shapes']))
        rows.append(
            f'<tr><td><code>{name}</code></td><td>{cls}</td>'
            f'<td style="text-align:right">{p:,}</td><td><code>{out}</code></td></tr>'
        )
    thead = ('<thead><tr><th>Module</th><th>Class</th>'
             '<th>Params (total)</th><th>Output shape</th></tr></thead>')
    return (
        '<table class="vlm-sortable" style="width:100%;border-collapse:collapse">'
        f'{thead}<tbody>'
        + ''.join(rows)
        + '</tbody></table>'
        + SORT_JS
    )


def _render_rst(card: dict[str, Any]) -> str:
    model_path = card['model_path']
    arch = card['architecture']
    total_params = _human_params(card['total_params'])
    fixture = card.get('fixture', {})
    device = card.get('device', 'cuda')
    modules = card['modules']

    title = model_path
    underline = '=' * len(title)

    tree_html = '\n'.join(_render_tree_html(_build_tree(modules)))
    table_html = _render_table_html(modules)

    # Indent raw HTML under `.. raw:: html` by 3 spaces.
    tree_block = '\n'.join('   ' + ln for ln in tree_html.splitlines())
    table_block = '\n'.join('   ' + ln for ln in table_html.splitlines())

    return f"""{title}
{underline}

- **Architecture:** ``{arch}``
- **HuggingFace:** `{model_path} <https://huggingface.co/{model_path}>`_
- **Total parameters:** {total_params} ({card['total_params']:,})
- **Probe fixture:** image ``{fixture.get('image', '?')}`` · prompt ``"{fixture.get('prompt', '')}"``
- **Device:** ``{device}``

.. note::

   Shapes below were captured during a single forward pass on the probe
   fixture. Modules whose output shape shows ``—`` were not invoked by that
   forward path (common for alternate branches, e.g. a vision-only model
   loaded with both text and image inputs).

Module tree
-----------

.. raw:: html

{tree_block}

Flat module list
----------------

Click a column header to sort. Module names are copy-pasteable into the
``modules:`` field of a YAML config.

.. raw:: html

{table_block}
"""


def _render_index(cards_by_arch: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        'Model Cards',
        '===========',
        '',
        'Auto-generated per-checkpoint cards showing the full module tree,',
        'parameter counts, and output shapes captured during a real forward',
        'pass. Pick a model below.',
        '',
        '.. toctree::',
        '   :maxdepth: 1',
        '',
    ]
    for arch in sorted(cards_by_arch):
        for card in sorted(cards_by_arch[arch], key=lambda c: c['model_path']):
            lines.append(f'   {card["model_path"]}')
    lines.append('')
    lines.append('By architecture')
    lines.append('---------------')
    lines.append('')
    for arch in sorted(cards_by_arch):
        lines.append(f'**{arch}**')
        lines.append('')
        for card in sorted(cards_by_arch[arch], key=lambda c: c['model_path']):
            mp = card['model_path']
            total = _human_params(card['total_params'])
            lines.append(f'- :doc:`{mp}` — {total} params')
        lines.append('')
    return '\n'.join(lines)


def build(cards_dir: Path = CARDS_DIR, out_dir: Path = OUT_DIR) -> int:
    """Render RST card pages from every JSON in ``cards_dir``.

    Args:
        cards_dir: Directory containing ``<provider>/<model>.json`` cards.
        out_dir: Directory to write ``<provider>/<model>.rst`` pages and the
            ``index.rst`` into.

    Returns:
        Number of card pages written.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    cards_dir = Path(cards_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    cards_by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for json_path in sorted(cards_dir.rglob('*.json')):
        with open(json_path) as f:
            card = json.load(f)
        rst = _render_rst(card)
        rel = json_path.relative_to(cards_dir).with_suffix('.rst')
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(rst)
        cards_by_arch[card['architecture']].append(card)
        written += 1
        logging.info(f'Rendered {out_path}')

    (out_dir / 'index.rst').write_text(_render_index(cards_by_arch))
    logging.info(f'Wrote index.rst with {written} card(s) across {len(cards_by_arch)} architecture(s).')
    return written


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cards-dir', type=Path, default=CARDS_DIR)
    p.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = p.parse_args()
    build(args.cards_dir, args.out_dir)
