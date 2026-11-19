"""Vector architecture diagram with schematic tokens and same-forward scoring."""

from __future__ import annotations

from matplotlib.patches import Polygon


def make_method(ctx):
    """Draw the exact GeoGate execution, without measured attention imagery."""
    fig = ctx.plt.figure(figsize=(ctx.TEXT_WIDTH, 3.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    def txt(x, y, value, size=7.5, color=None, **kwargs):
        return ax.text(x, y, value, fontsize=size, fontfamily='Times New Roman',
                       color=color or ctx.INK, **kwargs)

    def arrow(start, end, color=None, lw=1.0):
        ax.annotate('', xy=end, xytext=start, arrowprops={
            'arrowstyle': '-|>', 'color': color or ctx.MUTED,
            'lw': lw, 'mutation_scale': 9, 'shrinkA': 0, 'shrinkB': 0})

    def line(points, color=None, lw=1.0, **kwargs):
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=color or ctx.MUTED, lw=lw, **kwargs)

    def rounded(x, y, w, h, edge, fill, radius=.009, lw=.8):
        ax.add_patch(ctx.FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={radius}',
            facecolor=fill, edgecolor=edge, linewidth=lw))

    # (a) All 32 zero-indexed blocks, with the exact input/output boundary.
    txt(.035, .967, '(a)  Late pruning schedule', 9, weight='bold', va='top')
    txt(.965, .962, '32 decoder blocks | zero-indexed', 7.5,
        ctx.MUTED, ha='right', va='top')
    start, step, gap = .055, .900 / 32, .0024
    phases = ((0, 1, 'Absorption', '#EAF0F6'),
              (2, 13, 'Interaction', '#F7F3EA'),
              (14, 26, 'Readout', '#EAF5F1'),
              (27, 31, 'Tail', '#F0F2F5'))
    for low, high, label, fill in phases:
        center = start + (low + high + 1) * step / 2
        txt(center, .901, label, 7.5, ha='center')
        for layer in range(low, high + 1):
            ax.add_patch(ctx.Rectangle((start + layer * step, .842),
                step-gap, .041,
                facecolor=ctx.TEAL if layer == 14 else fill,
                edgecolor=ctx.TEAL if layer == 14 else '#AFBDC7', lw=.5))
    cut = start + 15 * step - gap / 2
    line([(cut, .832), (cut, .893)], ctx.CORAL, .95, ls=(0, (3, 2)))
    for layer in (0, 2, 8, 14, 20, 26, 31):
        txt(start + (layer+.5)*step-gap/2, .820, f'L{layer}',
            7.5, ctx.TEAL if layer == 14 else ctx.MUTED,
            ha='center', va='top')
    txt(.055, .768, 'Full sequence through L14', 8,
        ctx.BLUE, weight='bold')
    txt(.535, .768, 'Prune before L15', 8, ctx.TEAL, weight='bold')
    arrow((.357, .778), (.493, .778), ctx.MUTED, .8)
    line([(.035, .735), (.965, .735)], '#DCE3E8', .65)

    # (b) The primary architecture is a data-flow graph, with a scoring branch.
    txt(.035, .711, '(b)  One full-length forward, then attention-guided selection',
        9, weight='bold', va='top')
    txt(.120, .655, 'Input tokens', 8, weight='bold', ha='center')
    txt(.316, .655, 'Full-length prefill', 8, ctx.BLUE,
        weight='bold', ha='center')
    txt(.482, .655, 'Score once', 8, ctx.TEAL,
        weight='bold', ha='center')
    txt(.699, .655, 'Select', 8, ctx.CORAL,
        weight='bold', ha='center')
    txt(.913, .655, 'Continue', 8, ctx.TEAL,
        weight='bold', ha='center')

    selected = {3, 15}
    for i in range(20):
        ax.add_patch(ctx.Rectangle(
            (.041 + (i % 10)*.0156, .587 - (i // 10)*.025),
            .012, .016, facecolor=ctx.TEAL if i in selected else ctx.BLUE,
            edgecolor='white', linewidth=.25))
    for i in range(4):
        ax.add_patch(ctx.Rectangle((.041+i*.0198, .522), .0145, .017,
            facecolor='#D5DDE4', edgecolor='#8695A1', linewidth=.45))
    txt(.128, .529, 'text', 7.5, ctx.MUTED, va='center')
    txt(.120, .488, '20 spatial / all text', 7.5, ctx.MUTED,
        ha='center', va='top')

    # A stacked decoder module provides the visual architecture vocabulary.
    for dx, dy in ((.009, .010), (.0045, .005)):
        rounded(.260+dx, .515+dy, .113, .112,
                '#B7C8D6', '#EEF3F8', .006, .6)
    rounded(.260, .515, .113, .112, ctx.BLUE, '#EDF3F9', .006)
    txt(.3165, .587, 'Decoder', 8, ctx.BLUE, ha='center', va='center')
    txt(.3165, .547, 'L0-13', 8, ctx.BLUE, ha='center', va='center', weight='bold')
    arrow((.204, .574), (.248, .574), ctx.BLUE)

    rounded(.428, .515, .108, .112, ctx.TEAL, '#E7F4EF', .006)
    txt(.482, .586, 'L14', 9, ctx.TEAL, ha='center', va='center', weight='bold')
    txt(.482, .546, 'Full length', 7.5, ctx.TEAL, ha='center', va='center')
    arrow((.383, .574), (.418, .574), ctx.BLUE)
    txt(.482, .487, 'Enable eager before L14', 7.5,
        ctx.TEAL, ha='center', va='top')

    # The gate acts on the full L14 output; all text is included in the index map.
    ax.add_patch(Polygon([(.646, .572), (.698, .632), (.750, .572), (.698, .512)],
                         closed=True, facecolor='#FFF1E5',
                         edgecolor=ctx.CORAL, linewidth=.85))
    txt(.698, .572, 'Top-K', 8, ctx.CORAL, weight='bold', ha='center', va='center')
    arrow((.546, .574), (.636, .574), ctx.BLUE, 1.1)
    txt(.591, .606, 'L14 output', 7.5, ctx.MUTED, ha='center')

    for i in range(2):
        ax.add_patch(ctx.Rectangle((.777+i*.0208, .583), .0155, .021,
            facecolor=ctx.TEAL, edgecolor='white', linewidth=.25))
    for i in range(4):
        ax.add_patch(ctx.Rectangle((.777+i*.0198, .541), .0145, .017,
            facecolor='#D5DDE4', edgecolor='#8695A1', linewidth=.45))
    txt(.815, .497, '10% spatial', 7.5, ctx.TEAL, ha='center', va='top')
    txt(.815, .464, '+ all text', 7.5, ctx.MUTED, ha='center', va='top')
    arrow((.752, .574), (.770, .574), ctx.TEAL, 1.1)
    rounded(.873, .515, .095, .112, ctx.TEAL, '#E7F4EF', .006)
    txt(.9205, .587, 'Decoder', 8, ctx.TEAL, ha='center', va='center')
    txt(.9205, .546, 'L15-31', 8, ctx.TEAL, ha='center', va='center', weight='bold')
    arrow((.852, .574), (.865, .574), ctx.TEAL, 1.1)

    # Schematic query/key matrix: uniform cells carry no invented magnitudes.
    mx, my, cw, ch = .430, .327, .0130, .0200
    for row in range(5):
        for col in range(8):
            face = '#E7EDF2'
            if row == 0:
                face = '#F9DCC2' if col < 6 else '#D5DDE4'
            ax.add_patch(ctx.Rectangle((mx+col*cw, my+row*ch), cw-.001, ch-.001,
                facecolor=face, edgecolor='white', linewidth=.15))
    ax.add_patch(ctx.Rectangle((mx-.002, my-.002), 8*cw+.003, ch+.003,
        facecolor='none', edgecolor=ctx.CORAL, linewidth=.7))
    arrow((.482, .452), (.482, .433), ctx.TEAL, .85)
    txt(.413, .377, 'Attention', 7.5, ctx.MUTED, ha='right', va='center')
    txt(.413, .343, 'Last prompt', 7.5, ctx.CORAL, ha='right', va='center')
    txt(.558, .411, 'Spatial keys only', 7.5, ctx.CORAL, va='center')
    txt(.558, .377, 'Head mean (FastV)', 7.5, ctx.CORAL, va='center')
    txt(.558, .343, 'Same forward; no replay', 7.5, ctx.MUTED, va='center')
    # The spatial score row feeds the gate through the vertical branch.
    line([(.536, .337), (.548, .337), (.548, .304), (.744, .304),
          (.744, .476), (.698, .476), (.698, .489)],
         ctx.CORAL, .85)
    arrow((.698, .489), (.698, .503), ctx.CORAL, .85)
    txt(.035, .277,
        'Token cells and attention entries are schematic; original token order is retained.',
        7.5, ctx.MUTED, va='top')
    line([(.035, .237), (.965, .237)], '#DCE3E8', .65)

    # (c) Compact bookkeeping modules are subordinate to the architecture.
    txt(.035, .215, '(c)  Generation bookkeeping', 9, weight='bold', va='top')
    for x, w, title, detail, color, fill in (
        (.035, .281, 'Original position IDs', 'Preserve RoPE positions', ctx.BLUE, '#F1F5F9'),
        (.365, .281, 'KV cache synchronization', 'Index map + length snapshots', ctx.TEAL, '#EFF7F4'),
        (.695, .270, 'One-token decode', 'Mask uses true cache length', ctx.CORAL, '#FCF5ED'),
    ):
        rounded(x, .062, w, .096, '#CDD7DE', fill, .006, .6)
        txt(x+w/2, .126, title, 8, color, weight='bold', ha='center', va='center')
        txt(x+w/2, .088, detail, 7.5, ctx.MUTED, ha='center', va='center')
    arrow((.324, .111), (.354, .111), ctx.MUTED, .8)
    arrow((.654, .111), (.684, .111), ctx.MUTED, .8)
    return fig
