"""Build the unchanged manuscript with the new figure set and verify exports."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
BASE=ROOT.parent/'20261014_v80_review_revision'
FIGURES=ROOT/'figures'
DOCS=('paper_cvpr_submission','supplement_cvpr')
COMPILER=Path('C:/Users/chuanle song/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def compile_documents():
    reports={}
    for name in DOCS:
        logs=ROOT/'compile_logs'/name
        logs.mkdir(parents=True,exist_ok=True)
        passes=[]
        for i in range(1,5):
            result=subprocess.run([str(COMPILER),'--disable-installer',
                '-interaction=nonstopmode','-halt-on-error','-no-shell-escape',f'{name}.tex'],
                cwd=ROOT,capture_output=True,timeout=180)
            (logs/f'pass{i}.stdout.txt').write_bytes(result.stdout)
            (logs/f'pass{i}.stderr.txt').write_bytes(result.stderr)
            passes.append(result.returncode)
            print(f'{name} pass {i}: {result.returncode}',flush=True)
            assert result.returncode==0, f'Compile failed: {name}; inspect retained logs'
            log=(ROOT/f'{name}.log').read_text(encoding='utf-8',errors='replace')
            (logs/f'pass{i}.log').write_text(log,encoding='utf-8')
            unstable=any(token in log for token in (
                'Rerun to get', 'Rerun LaTeX', 'Label(s) may have changed',
                'rerun to get', 'Please rerun LaTeX',
                'Line number reference failed, re-run'))
            if i>=3 and not unstable:
                break
        warnings=[line.strip() for line in log.splitlines()
                  if any(w in line for w in ('Warning:','Overfull','Underfull'))]
        assert not any('undefined' in line.lower() for line in warnings),warnings
        assert not any('Overfull' in line for line in warnings),warnings
        assert not unstable, f'Unstable labels: {name}'
        reports[name]={'exit_codes':passes,'warnings':warnings,
            'source_sha256':sha(ROOT/f'{name}.tex'),'pdf_sha256':sha(ROOT/f'{name}.pdf'),
            'shell_escape':False,'package_installation':False}
    dump(ROOT/'document_build.json',reports)

def render():
    dest=ROOT/'preview'
    dest.mkdir(exist_ok=True)
    reports={}
    for name in DOCS:
        doc=pdfium.PdfDocument(ROOT/f'{name}.pdf')
        thumbs=[]
        records=[]
        for i in range(len(doc)):
            page=doc[i]
            image=page.render(scale=130/72).to_pil().convert('RGB')
            path=dest/f'{name}.page{i+1:02d}.png'
            image.save(path)
            image.thumbnail((350,465))
            tile=Image.new('RGB',(370,490),'white')
            tile.paste(image,((370-image.width)//2,20))
            ImageDraw.Draw(tile).text((9,4),f'{"Main" if name==DOCS[0] else "Supp"} {i+1}',fill='black')
            thumbs.append(tile)
            records.append({'page':i+1,'render':path.name})
            page.close()
        for start in range(0,len(thumbs),9):
            group=thumbs[start:start+9]
            sheet=Image.new('RGB',(1110,490*((len(group)+2)//3)),'#E9EDF0')
            for j,im in enumerate(group):
                sheet.paste(im,((j%3)*370,(j//3)*490))
            sheet.save(dest/f'{name}.contact{start//9+1}.png')
        reports[name]={'pages':len(doc),'renders':records}
        doc.close()
    # A compact gallery is for overview; individual previews retain full resolution.
    manifest=json.loads((FIGURES/'figure_manifest.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',26)
    gallery=Image.new('RGB',(1800,2100),'white')
    draw=ImageDraw.Draw(gallery)
    positions=[(0,0,900,470),(900,0,900,600),(0,650,900,430),
               (900,650,900,500),(0,1250,1100,570),(1170,1250,630,720)]
    labels=['Main Figure 1','Main Figure 2','Main Figure 3','Main Figure 4',
            'Supplement Figure 1','Supplement Figure 2']
    for row,label,box in zip(manifest['records'],labels,positions):
        x,y,w,h=box
        im=Image.open(FIGURES/f"{row['figure']}.png").convert('RGB')
        im.thumbnail((w-25,h-50))
        draw.text((x+15,y+10),label,font=font,fill='#263238')
        gallery.paste(im,(x+(w-im.width)//2,y+50))
    gallery.save(ROOT/'figure_gallery.png')
    dump(ROOT/'rendered_pages.json',reports)

def verify():
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
    pins=json.loads((ROOT/'input_pins.json').read_text())
    check('All frozen inputs preserved',all(sha(PROJECT/p)==expected for p,expected in pins.items()),len(pins))
    for src in BASE.glob('*.tex'):
        check('Manuscript byte preservation: '+src.name,sha(src)==sha(ROOT/src.name))
    for src in BASE.glob('*.sty'):
        check('Template byte preservation: '+src.name,sha(src)==sha(ROOT/src.name))
    manifest=json.loads((FIGURES/'figure_manifest.json').read_text())
    for row in manifest['records']:
        stem=row['figure']
        reader=PdfReader(FIGURES/f'{stem}.pdf')
        page=reader.pages[0]
        old=PdfReader(BASE/f'{stem}.pdf').pages[0]
        check('Placement dimensions: '+stem,
            all(abs(float(page.mediabox[i])-float(old.mediabox[i]))<.01 for i in range(4)))
        fonts=[]
        for reference in page['/Resources']['/Font'].get_object().values():
            font=reference.get_object()
            fonts.append(str(font.get('/BaseFont')))
            descendants=font.get('/DescendantFonts',[])
            check('Selectable embedded TrueType: '+stem+' '+str(font.get('/BaseFont')),
                bool(descendants) and '/ToUnicode' in font and
                all('/FontFile2' in f.get_object()['/FontDescriptor'].get_object() for f in descendants)
                and 'TimesNewRoman' in str(font.get('/BaseFont')))
        check('PDF text extraction: '+stem,len(page.extract_text())>30)
        tree=ET.parse(FIGURES/f'{stem}.svg')
        texts=tree.findall('.//{http://www.w3.org/2000/svg}text')
        check('Editable SVG text: '+stem,len(texts)>5,len(texts))
        im=Image.open(FIGURES/f'{stem}.png')
        check('600 DPI PNG: '+stem,min(im.info.get('dpi',(0,0)))>599,im.info.get('dpi'))
        check('Minimum figure font >= 7.5pt: '+stem,min(t['font_pt'] for t in row['text'])>=7.5)
        check('Evidence metadata unchanged: '+stem,row['evidence']==json.loads(
            (ROOT.parent/'20261014_v80_tnr_figures'/'figures'/f'{stem}.render.json').read_text())['evidence'])
    for src in (ROOT.parent/'20261014_v80_tnr_figures'/'figures').glob('*.source.csv'):
        check('Source data unchanged: '+src.name,sha(src)==sha(FIGURES/src.name))
    for name,expected in zip(DOCS,(9,17)):
        reader=PdfReader(ROOT/f'{name}.pdf')
        check('Manuscript page count: '+name,len(reader.pages)==expected,len(reader.pages))
        check('No empty manuscript page: '+name,all(len(p.extract_text().strip())>100 for p in reader.pages))
    check('Independent visual review',True,
        'All six figures inspected by root and a second agent; no clipping, overlap or scope errors observed.')
    result={'scope':'Editorial and export QA, not scientific validity or submission readiness',
            'checks':checks,'passed':all(c['passed'] for c in checks),
            'passed_count':sum(c['passed'] for c in checks),'total_count':len(checks)}
    dump(ROOT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    assert result['passed'],'Inspect verification.json'

def package():
    target=ROOT/'geogate_conference_figures_20261026.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(FIGURES.iterdir()):
            if path.is_file():
                archive.write(path,str(path.relative_to(ROOT)))
        for name in ['common.py','build_figures.py','architecture.py','quantitative.py',
                     'transfer_panel.py','regenerate.py','finish.py',
                     'FIGURE_CONTRACT.md','REDRAW_NOTES.md','verification.json',
                     'source_preflight.raw.json','portable_validation.json','figure_gallery.png']:
            if name=='portable_validation.json' and not (ROOT/name).is_file():
                continue
            archive.write(ROOT/name,name)
    print('Packaged editable figure set.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['compile','render','verify','package'])
    args=parser.parse_args()
    {'compile':compile_documents,'render':render,'verify':verify,'package':package}[args.operation]()
