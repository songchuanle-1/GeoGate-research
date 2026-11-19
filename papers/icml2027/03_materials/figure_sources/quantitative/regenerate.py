"""Redraw the portable conference figure bundle without the original project."""
import shutil
import common as c
import build_figures as b
import quantitative as q
import architecture as a

def main():
    c.DATA=c.ROOT/'figures'
    b.OUT=c.ROOT/'regenerated'
    b.OUT.mkdir(exist_ok=False)
    c.setup()
    functions=(q.profiles,a.make_method,q.paired,q.transfer,q.overview,q.historical)
    records=[b.export(fn(c),stem) for fn,stem in zip(functions,b.STEMS)]
    for source in c.DATA.glob('*.source.csv'):
        shutil.copy2(source,b.OUT/source.name)
    b.write_json(b.OUT/'figure_manifest.json',{'records':records,'new_statistics':False})
    print('Redrew all six figures from the included frozen evidence.')

if __name__=='__main__':
    main()
