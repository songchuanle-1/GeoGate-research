"""Conference-style transfer forest and aligned frozen-statistics ledger."""

from __future__ import annotations


def make_transfer(ctx):
    """Draw all four frozen effects without introducing a new inference rule."""
    effects = ctx.evidence('fig_transfer_effects')['effects']
    assert len(effects) == 4
    fig = ctx.plt.figure(figsize=(ctx.TEXT_WIDTH, 3.2))

    def text(x, y, label, size=7.5, color=None, **kwargs):
        return fig.text(x, y, label, fontsize=size,
                        color=color or ctx.INK, **kwargs)

    def rule(x0, x1, y, color='#DCE3E8', lw=.6):
        fig.add_artist(ctx.plt.Line2D([x0, x1], [y, y],
                       transform=fig.transFigure, color=color, linewidth=lw))

    text(.035, .958, '(a)  SQA3D validation', 9, weight='bold', va='top')
    text(.655, .958, '3D-LLaVA', 7.5, ctx.MUTED, ha='right', va='top')
    text(.035, .864, 'Frozen fold rule: lower CI bound > -1 EM',
         7.5, ctx.CORAL)

    # The forest is the principal visual; the complete numbers form a compact
    # ledger below it, rather than separate three-line text blocks beside it.
    forest = fig.add_axes([.143, .469, .512, .337])
    forest.set(xlim=(-2.4, .6), ylim=(-.60, 2.60),
               xticks=[-2, -1, 0], xlabel='EM change (points)')
    forest.set_yticks([2, 1, 0], ['Fold A', 'Fold B', 'A+B*'])
    forest.spines['left'].set_visible(False)
    forest.tick_params(axis='y', length=0, pad=6)
    forest.axvline(0, color=ctx.MUTED, linewidth=.8)
    # Only the two fold rows have a frozen -1 non-inferiority reference.
    forest.plot([-1, -1], [.63, 2.42], color=ctx.CORAL,
                linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)
    for index, effect in enumerate(effects[:3]):
        y = 2 - index
        delta = effect['delta']
        lo, hi = effect['ci95']
        forest.hlines(y, -2.4, .6, color='#EDF1F4', linewidth=.6, zorder=0)
        forest.errorbar(delta, y, xerr=[[delta-lo], [hi-delta]],
                        fmt='o' if index < 2 else 'D',
                        color=ctx.BLUE if index < 2 else ctx.MUTED,
                        markersize=4.0, capsize=2.4,
                        linewidth=1.25, zorder=3)
    forest.grid(axis='x', color='#E7ECF0', linewidth=.5)
    forest.xaxis.label.set_size(7.5)

    # A readable three-column table keeps sample counts and exact estimates
    # aligned and makes clear that the combined row is descriptive.
    text(.035, .341, 'Cohort', 7.5, ctx.MUTED)
    text(.286, .341, 'Questions / scenes', 7.5, ctx.MUTED)
    text(.655, .341, 'EM change [95% CI]', 7.5, ctx.MUTED, ha='right')
    rule(.035, .655, .322)
    labels = ('A: disjoint scenes', 'B: overlapping scenes', 'Combined A+B*')
    for index, (label, effect) in enumerate(zip(labels, effects[:3])):
        y = .282 - index * .074
        lo, hi = effect['ci95']
        color = ctx.INK if index < 2 else ctx.MUTED
        text(.035, y, label, 7.5, color, va='center')
        text(.286, y, f"{effect['questions']:,} / {effect['scenes']}",
             7.5, color, va='center')
        text(.655, y, f"{effect['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]",
             7.5, color, ha='right', va='center')
        rule(.035, .655, y-.035, lw=.4)

    # The independent-domain result has its own horizontal scale and test.
    text(.722, .958, '(b)  MMScan / 3RScan', 8.5,
         weight='bold', va='top')
    text(.722, .864, 'LLaVA-3D', 7.5, ctx.MUTED)
    mm = effects[3]
    text(.722, .797, f"{mm['questions']:,} questions / {mm['scenes']} scenes", 7.5)
    right = fig.add_axes([.739, .524, .234, .215])
    right.set(xlim=(-.4, .3), ylim=(-1, 1), xticks=[-.3, 0, .3],
              xlabel='EM change (points)')
    right.set_yticks([])
    right.spines['left'].set_visible(False)
    right.axvline(0, color=ctx.MUTED, linewidth=.8)
    lo, hi = mm['ci95']
    right.errorbar(mm['delta'], 0,
                   xerr=[[mm['delta']-lo], [hi-mm['delta']]],
                   fmt='o', color=ctx.BLUE, markersize=4,
                   capsize=2.4, linewidth=1.25)
    right.grid(axis='x', color='#E7ECF0', linewidth=.5)
    right.xaxis.label.set_size(7.5)

    text(.722, .377, 'EM change [95% CI]', 7.5, ctx.MUTED)
    text(.722, .316, f"{mm['delta']:+.4f}", 9, ctx.BLUE, weight='bold')
    text(.722, .254, f'[{lo:+.4f}, {hi:+.4f}]', 7.5)
    text(.722, .190, 'No frozen loss margin', 7.5, ctx.MUTED)

    # A fine separator, rather than a box or card, distinguishes protocols.
    fig.add_artist(ctx.plt.Line2D([.689, .689], [.145, .932],
                   transform=fig.transFigure, color='#DCE3E8', linewidth=.65))
    text(.035, .066, 'Both folds: non-inferiority NOT established',
         7.5, ctx.CORAL, weight='bold')
    text(.722, .066, 'Superiority NOT established',
         7.5, ctx.CORAL, weight='bold')
    text(.035, .018,
         '*A+B is not independent. Marginal two-sided 95% scene-bootstrap CIs; different x scales.',
         7.5, ctx.MUTED)
    return fig
