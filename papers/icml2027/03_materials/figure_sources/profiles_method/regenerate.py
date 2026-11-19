"""Rebuild the two label-revised figures from included frozen evidence.

Run with a Python installation containing matplotlib/numpy on Windows, where
the existing Times New Roman files are loaded from C:/Windows/Fonts.
This is editorial rendering, not model execution or statistical recomputation.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import common as c
import profiles
import architecture
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'exports'
STEMS = ('fig1_three_phase', 'fig2_geogate_method')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        if not bbox.width or not bbox.height:
            continue
        name = Path(c.font_manager.findfont(text.get_fontproperties(),
                                            fallback_to_default=False)).name
        bounds = [bbox.x0 / fig.bbox.width, bbox.y0 / fig.bbox.height,
                  bbox.x1 / fig.bbox.width, bbox.y1 / fig.bbox.height]
        assert name.lower() in c.FONTS, (stem, text.get_text(), name)
        assert text.get_fontsize() >= 7.5, (stem, text.get_text())
        assert min(bounds) >= -.002 and max(bounds) <= 1.002, (stem, text.get_text(), bounds)
        rows.append({'text': text.get_text(), 'font_pt': text.get_fontsize(),
                     'font_file': name, 'bbox': bounds})
    stamp = datetime(2026, 11, 4, tzinfo=timezone.utc)
    fig.savefig(OUT / f'{stem}.pdf',
                metadata={'Creator': 'Matplotlib', 'CreationDate': stamp, 'ModDate': stamp})
    fig.savefig(OUT / f'{stem}.svg', metadata={'Date': '2026-11-04'})
    fig.savefig(OUT / f'{stem}.png', dpi=600)
    record = {'figure': stem, 'width_in': fig.get_figwidth(),
              'height_in': fig.get_figheight(), 'font': 'Times New Roman',
              'text': rows, 'evidence': c.evidence(stem),
              'new_statistics': False, 'excluded_observations': 0,
              'outputs': {ext: sha(OUT / f'{stem}.{ext}')
                          for ext in ('pdf', 'svg', 'png')}}
    (OUT / f'{stem}.render.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    c.plt.close(fig)
    return record


def main():
    OUT.mkdir(exist_ok=True)
    c.setup()
    records = [export(fn(c), stem) for fn, stem in
               zip((profiles.profiles, architecture.make_method), STEMS)]
    for stem in STEMS:
        shutil.copy2(OUT / f'{stem}.pdf', ROOT.parent / f'{stem}.pdf')
    (OUT / 'figure_manifest.json').write_text(
        json.dumps({'records': records, 'new_statistics': False,
                    'frozen_sources': {p.name: sha(p) for p in (ROOT / 'frozen').iterdir()},
                    'changes': 'Descriptive attention labels and explicit all-layer prompt-cache compaction.'},
                   indent=2)+'\n', encoding='utf-8')
    print('Rebuilt two figures from unchanged frozen evidence.')


if __name__ == '__main__':
    main()
