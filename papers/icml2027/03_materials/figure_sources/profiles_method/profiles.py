"""Conference-style quantitative panels; the included frozen evidence is exact."""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors

def header(ctx,fig,x,y,letter,label,scope=None):
    fig.text(x,y,f'({letter})  {label}',fontsize=9,weight='bold',va='top')
    if scope:
        fig.text(x,y-.065,scope,fontsize=7.5,color=ctx.MUTED,va='top')

def profiles(c):
    ev=c.evidence('fig1_three_phase')
    fig=c.plt.figure(figsize=(c.TEXT_WIDTH,2.8))
    x=c.np.arange(32)
    boxes=((.07,.30,.345,.47),(.51,.30,.205,.47),(.80,.30,.18,.47))
    definitions=(('Spatial attention','Last-prompt query','Spatial attention (%)',
                  (0,75),[0,25,50,75]),
                 ('Half-mass coverage','Last-prompt query','Coverage (%)',
                  (0,50),[0,25,50]),
                 ('Gini inequality','All-query attention','Gini coefficient',
                  (.2,.6),[.2,.4,.6]))
    handles=[]
    for i,(bounds,definition,series) in enumerate(zip(boxes,definitions,ev['series'])):
        ax=fig.add_axes(bounds)
        c.background(ax)
        name,scope,ylabel,ylim,yticks=definition
        h1,=ax.plot(x,series['llava3d'],color=c.BLUE,lw=1.45,
                    marker='o',ms=2.7,markevery=2,markeredgewidth=.35,markeredgecolor='white')
        h2,=ax.plot(x,series['3dllava'],color=c.CORAL,lw=1.45,ls=(0,(4,2)),
                    marker='s',ms=2.5,markevery=2,markeredgewidth=.3,markeredgecolor='white')
        if i==0:
            handles=[h1,h2]
        ax.set(xlim=(-.5,31.5),ylim=ylim,yticks=yticks,xticks=[0,8,14,24,31],
               ylabel=ylabel,xlabel='Layer (zero-indexed)')
        ax.grid(axis='y')
        ax.tick_params(pad=3)
        header(c,fig,bounds[0],.91,'abc'[i],name,scope)
        for tick in ax.get_xticklabels():
            if tick.get_text()=='14':
                tick.set_color(c.TEAL)
                tick.set_fontweight('bold')
    inset=fig.add_axes([.158,.555,.244,.187])
    for lo,hi,_,color in c.PHASES:
        inset.axvspan(lo,hi,color=color,alpha=.48,lw=0)
    inset.axvline(14,color=c.TEAL,ls=(0,(3,2)),lw=.75)
    for key,color,ls in [('llava3d',c.BLUE,'-'),('3dllava',c.CORAL,(0,(4,2)))]:
        inset.plot(x[2:],ev['series'][0][key][2:],color=color,lw=1.1,ls=ls)
    inset.set(xlim=(1.5,31.5),ylim=(0,25),xticks=[2,14,31],yticks=[0,20])
    inset.set_title('L2-31 detail (same points)',fontsize=7.5,loc='left',pad=3)
    inset.tick_params(labelsize=7.5,length=2,pad=1)
    inset.grid(axis='y',lw=.4)
    for s in inset.spines.values():
        s.set_visible(True)
        s.set_color('#AFBBC4')
        s.set_linewidth(.55)
    fig.legend(handles,['LLaVA-3D | 1497 spatial tokens','3D-LLaVA | 100 superpoint tokens'],
               loc='upper center',bbox_to_anchor=(.5,1.004),ncol=2,
               handlelength=2.7,columnspacing=2.5,fontsize=8)
    fig.legend([Patch(facecolor=p[3],edgecolor='#BAC5CE',lw=.6) for p in c.PHASES],
               [p[2] for p in c.PHASES],loc='lower center',bbox_to_anchor=(.5,.070),
               ncol=4,fontsize=7.5,handlelength=1,columnspacing=1.5)
    fig.text(.5,.027,'Descriptive attention regions | Original 50-scene profiles | No uncertainty bands',
             fontsize=7.5,color=c.MUTED,ha='center')
    return fig

