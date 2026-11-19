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
    definitions=(('Spatial readout','Last-prompt query','Spatial attention (%)',
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
    fig.text(.5,.027,'Descriptive phases | Original 50-scene profiles | No uncertainty bands',
             fontsize=7.5,color=c.MUTED,ha='center')
    return fig

def paired(c):
    ev=c.evidence('fig_layer_replay_paired_final')
    means,contrasts=ev['arm_em_percent'],ev['contrasts']
    layers=[2,4,6,8,10,12,13,14,15,16,18]
    fig=c.plt.figure(figsize=(c.TEXT_WIDTH,2.48*c.TEXT_WIDTH/7.16))
    ax=fig.add_axes([.068,.24,.415,.58])
    ax.plot(layers,[means[f'L{n}'] for n in layers],lw=1.5,color=c.BLUE,
            marker='o',ms=3.4,markeredgecolor='white',markeredgewidth=.5)
    ax.axhline(means['native'],color=c.MUTED,ls=(0,(3,2)),lw=.95)
    ax.axvspan(13.6,14.4,color='#E4F3EE',lw=0,zorder=0)
    ax.scatter([14],[means['L14']],s=30,color=c.TEAL,edgecolor='white',lw=.6,zorder=5)
    ax.annotate('L14  27.7219',xy=(14,means['L14']),xytext=(10.3,28.55),
                fontsize=7.5,color=c.TEAL,weight='bold',
                arrowprops={'arrowstyle':'-','color':c.TEAL,'lw':.8})
    ax.text(2.1,26.98,'Native  27.4225',fontsize=7.5,color=c.MUTED)
    ax.set(xlim=(1.5,18.5),ylim=(21.9,29.1),xticks=[2,6,10,14,18],yticks=[22,24,26,28],
           xlabel='Pruning block (zero-indexed)',ylabel='ScanQA EM (%)')
    ax.grid(axis='y')
    header(c,fig,.068,.94,'a','Paired-replay means')
    ax=fig.add_axes([.615,.24,.365,.58])
    for i,row in enumerate(contrasts):
        if i in [0,8,9,10]:
            ax.axhspan(i-.46,i+.46,color='#F1F4F6',lw=0,zorder=0)
        point=row['delta_em_points']
        lo,hi=row['max_t_95ci_em_points']
        resolved=row['max_t_decision']=='positive'
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],
                    fmt='o' if resolved else 's',color=c.TEAL if resolved else c.MUTED,
                    ms=3.5,capsize=2,lw=1.1)
    ax.axvline(0,color=c.MUTED,ls=(0,(3,2)),lw=.8)
    ax.set(xlim=(-.8,7.5),ylim=(10.7,-.7),xticks=[0,2,4,6],
           xlabel='L14 minus comparator (EM points)')
    ax.set_yticks(range(11),[r['comparator'] for r in contrasts])
    ax.grid(axis='x')
    header(c,fig,.615,.94,'b','Simultaneous 95% max-t CIs')
    fig.text(.068,.027,'ScanQA | 4,675 questions / 71 scenes | R = 0.10',fontsize=7.5,color=c.MUTED)
    fig.legend([Line2D([],[],color=c.TEAL,marker='o',lw=1),
                Line2D([],[],color=c.MUTED,marker='s',lw=1)],
               ['Resolved positive','Unresolved'],loc='lower right',
               bbox_to_anchor=(.99,.003),ncol=2,handlelength=1.3,columnspacing=1.1,fontsize=7.5)
    return fig

def transfer(c):
    from transfer_panel import make_transfer
    return make_transfer(c)

def overview(c):
    ev=c.evidence('fig_teaser')
    fig=c.plt.figure(figsize=(c.TEXT_WIDTH,2.85))
    ax=fig.add_axes([.105,.38,.235,.36])
    schedules=ev['configured_retention_percent']
    matrix=c.np.array([schedules[k] for k in ('early','progressive','geogate')])
    image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='Blues',interpolation='nearest')
    ax.set(xticks=[0,8,14,24,31],xlabel='Block input (zero-indexed)')
    ax.set_yticks([0,1,2],['Early','3-stage','GeoGate'])
    ax.tick_params(axis='y',length=0,pad=4)
    for y in [.5,1.5]:
        ax.axhline(y,color='white',lw=3)
    ax.axvline(14.5,color=c.CORAL,lw=1,ls=(0,(2,2)))
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Every cell is one exact configured-retention value, not a measured attention value.
    bar=fig.add_axes([.105,.23,.235,.027])
    cb=fig.colorbar(image,cax=bar,orientation='horizontal',ticks=[0,50,100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2,labelsize=7.5,pad=2)
    cb.set_label('Spatial tokens retained (%)',size=7.5,labelpad=2)
    header(c,fig,.045,.93,'a','Retention schedules','Configured, not measured')
    fig.text(.045,.03,'Early: L2 / 10% | 3-stage: 12.5%\nGeoGate: L14 / 10%',
             fontsize=7.5,color=c.MUTED,linespacing=1.5)
    ax=fig.add_axes([.455,.315,.255,.43])
    c.background(ax)
    x=c.np.arange(32)
    ax.plot(x,ev['attention_percent'],color=c.BLUE,lw=1.4)
    ax.scatter([14],[ev['attention_percent'][14]],s=26,color=c.TEAL,
               edgecolor='white',lw=.6,zorder=5)
    ax.annotate('Score L14\nPrune before L15',xy=(14,ev['attention_percent'][14]),
                xytext=(8.5,47),fontsize=7.5,color=c.TEAL,linespacing=1.35,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.4},
                arrowprops={'arrowstyle':'->','color':c.TEAL,'lw':.8})
    ax.set(xlim=(-.5,31.5),ylim=(0,75),xticks=[0,14,31],yticks=[0,25,50,75],
           xlabel='Layer (zero-indexed)',ylabel='Spatial attention (%)')
    ax.grid(axis='y')
    header(c,fig,.455,.93,'b','Spatial attention','Last-prompt query')
    fig.text(.455,.075,'LLaVA-3D | 50 scenes\nOriginal profile; no CIs',
             fontsize=7.5,color=c.MUTED,linespacing=1.5)
    header(c,fig,.775,.93,'c','Cost scopes')
    # Separate quantities have no shared numeric axis or combined performance claim.
    for y,scope,value,detail,color in [
        (.71,'CONFIGURED','90%','Spatial tokens removed',c.TEAL),
        (.455,'ANALYTICAL','42.1%','LM FLOPs saved',c.BLUE),
        (.20,'MEASURED','1.061-1.072x','Generation-core speedup',c.CORAL)]:
        fig.text(.775,y+.065,scope,fontsize=7.5,color=c.MUTED,weight='bold')
        fig.text(.775,y-.025,value,fontsize=13,weight='bold',color=color)
        fig.text(.775,y-.087,detail,fontsize=7.5)
        fig.add_artist(Line2D([.775,.975],[y-.12,y-.12],transform=fig.transFigure,
                              lw=.6,color='#DDE4E9'))
    fig.text(.775,.025,'H100, B1 | Prep. excluded',fontsize=7.5,color=c.MUTED)
    return fig

def historical(c):
    ev=c.evidence('fig_layer_curve')
    x,y,base=ev['layers'],ev['em'],ev['baseline']
    fig=c.plt.figure(figsize=(c.COLUMN_WIDTH,2.7))
    ax=fig.add_axes([.17,.235,.795,.53])
    ax.axvspan(13.55,14.45,color='#E4F3EE',lw=0,zorder=0)
    ax.axhline(base,color=c.MUTED,ls=(0,(3,2)),lw=.9)
    ax.plot(x,y,color=c.BLUE,lw=1.5,marker='o',ms=4,
            markeredgecolor='white',markeredgewidth=.6)
    ax.scatter([14],[y[x.index(14)]],s=36,color=c.TEAL,edgecolor='white',lw=.6,zorder=5)
    ax.text(2,27.66,'Native 27.34',fontsize=7.5,color=c.MUTED)
    ax.annotate('L14  27.79\nSelected setting',xy=(14,27.79),xytext=(9.5,24.35),
                fontsize=7.5,color=c.TEAL,weight='bold',linespacing=1.4,
                arrowprops={'arrowstyle':'->','lw':.8,'color':c.TEAL})
    ax.set(xlim=(1.3,18.7),ylim=(21.5,29),xticks=[2,6,10,14,18],yticks=[22,24,26,28],
           xlabel='Pruning layer (after block)',ylabel='ScanQA EM@1 (%)')
    ax.grid(axis='y')
    fig.text(.17,.95,'Historical layer sweep',fontsize=9.5,weight='bold',va='top')
    fig.text(.17,.865,'4,675 questions | 10% spatial retention',fontsize=7.5,color=c.MUTED,va='top')
    fig.text(.17,.045,'Original aggregate values | No CIs',fontsize=7.5,color=c.MUTED)
    return fig
