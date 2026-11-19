"""Build the conference-style vector figures using only frozen evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import common as c
from common import plt
from matplotlib.text import Text

ROOT, PROJECT, BASE, DATA = c.ROOT, c.PROJECT, c.BASE, c.DATA
STEMS=('fig1_three_phase','fig2_geogate_method','fig_layer_replay_paired_final',
       'fig_transfer_effects','fig_teaser','fig_layer_curve')
OUT=ROOT/'figures'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def pin_inputs():
    paths=set(DATA.glob('*.source.csv')) | set(DATA.glob('*.render.json'))
    paths |= set(BASE.glob('*.tex')) | set(BASE.glob('*.sty')) | set(BASE.glob('*.pdf'))
    previous=ROOT.parent/'20261026_v80_figure_polish'
    paths |= set((previous/'figures').glob('*.*'))
    paths |= {PROJECT/p for p in ('paper/state.json','FOLLOWUP_EXPERIMENTS.md',
        'docs/paper_cvpr_submission.tex','docs/supplement_cvpr.tex',
        'docs/paper_cvpr_submission.pdf','docs/supplement_cvpr.pdf',
        'results/raw/e1_profile_llava3d.json','results/raw/e1b_profile_3dllava.json',
        'artifacts/geogate_cvpr_submission_v80.zip')}
    pins={str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in sorted(paths)}
    target=ROOT/'input_pins.json'
    if target.exists():
        assert json.loads(target.read_text())==pins,'Frozen inputs changed'
    else:
        write_json(target,pins)
    return pins

def export(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    rows=[]
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb=t.get_window_extent(renderer)
        if not bb.width or not bb.height:
            continue
        name=Path(c.font_manager.findfont(t.get_fontproperties(),fallback_to_default=False)).name
        bounds=[bb.x0/fig.bbox.width,bb.y0/fig.bbox.height,bb.x1/fig.bbox.width,bb.y1/fig.bbox.height]
        assert name.lower() in c.FONTS,(stem,t.get_text(),name)
        assert t.get_fontsize()>=7.5,(stem,t.get_text(),t.get_fontsize())
        assert min(bounds)>=-.002 and max(bounds)<=1.002,(stem,t.get_text(),bounds)
        rows.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'font_file':name,'bbox':bounds})
    stamp=datetime(2026,10,26,tzinfo=timezone.utc)
    fig.savefig(OUT/f'{stem}.pdf',metadata={'Creator':'Matplotlib','CreationDate':stamp,'ModDate':stamp})
    fig.savefig(OUT/f'{stem}.svg',metadata={'Date':'2026-10-26'})
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    record={'figure':stem,'width_in':fig.get_figwidth(),'height_in':fig.get_figheight(),
            'font':'Times New Roman','text':rows,'evidence':c.evidence(stem),
            'new_statistics':False,'excluded_observations':0,
            'outputs':{ext:sha(OUT/f'{stem}.{ext}') for ext in ('pdf','svg','png')}}
    write_json(OUT/f'{stem}.render.json',record)
    plt.close(fig)
    return record

def main():
    pins=pin_inputs()
    c.setup()
    import quantitative as q
    import architecture as a
    OUT.mkdir(exist_ok=True)
    functions=(q.profiles,a.make_method,q.paired,q.transfer,q.overview,q.historical)
    records=[]
    for stem,fn in zip(STEMS,functions):
        records.append(export(fn(c),stem))
        print(f'Redrawn {stem}',flush=True)
    for source in DATA.glob('*.source.csv'):
        shutil.copy2(source,OUT/source.name)
    raw=[json.loads((PROJECT/p).read_text()) for p in
         ('results/raw/e1_profile_llava3d.json','results/raw/e1b_profile_3dllava.json')]
    for series in records[0]['evidence']['series']:
        for key,source,field in zip(('llava3d','3dllava'),raw,series['keys']):
            assert series[key]==[r[field]*series['scale'] for r in source['layers']]
    for p,expected in pins.items():
        assert sha(PROJECT/p)==expected,p
    for extension in ('*.tex','*.sty'):
        for source in BASE.glob(extension):
            shutil.copy2(source,ROOT/source.name)
    for stem in STEMS:
        shutil.copy2(OUT/f'{stem}.pdf',ROOT/f'{stem}.pdf')
    write_json(OUT/'figure_manifest.json',{'schema':'geogate-conference-redraw-v1',
        'backend':'Python/matplotlib','figure_count':6,'records':records,
        'raw_attention_values_verified':192,'new_scientific_evidence':False,
        'sources_unchanged':True,'excluded_observations':0,
        'scripts':{name:sha(ROOT/name) for name in
                   ('common.py','architecture.py','quantitative.py','transfer_panel.py','build_figures.py')}})

if __name__=='__main__':
    main()
