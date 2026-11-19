"""Shared print-scale style and frozen evidence for the editorial figures."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = ROOT.parent / '20261014_v80_review_revision'
DATA = ROOT.parent / '20261014_v80_tnr_figures' / 'figures'
TEXT_WIDTH, COLUMN_WIDTH = 6.875, 3.28125
BLUE, TEAL, CORAL = '#275D8C', '#16877D', '#D27C43'
INK, MUTED = '#1F2933', '#627282'
PHASES = ((-.5, 1.5, 'Absorption L0-1', '#EAF0F6'),
          (1.5, 13.5, 'Interaction L2-13', '#F7F4ED'),
          (13.5, 26.5, 'Readout L14-26', '#EAF4F1'),
          (26.5, 31.5, 'Tail L27-31', '#F3F4F5'))
FONTS = ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf')

def setup():
    for name in FONTS:
        font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts') / name))
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.serif': ['Times New Roman'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#AAB7BD',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'axes.axisbelow': True,
        'grid.color': '#E6EAED', 'grid.linewidth': .5,
        'xtick.major.size': 2.7, 'ytick.major.size': 2.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'geogate-conference-redraw-20261026',
        'savefig.facecolor': 'white', 'legend.frameon': False,
    })

def evidence(stem):
    path = DATA / f'{stem}.render.json'
    if not path.exists():
        path = ROOT / 'figures' / f'{stem}.render.json'
    return json.loads(path.read_text(encoding='utf-8'))['evidence']

def background(ax):
    for lo, hi, _, color in PHASES:
        ax.axvspan(lo, hi, color=color, alpha=.40, lw=0, zorder=0)
        ax.axvspan(lo, hi, ymin=.96, ymax=1, color=color, lw=0, zorder=1)
    ax.axvline(14, color=TEAL, ls=(0, (3, 2)), lw=.85, zorder=2)

def title(ax, letter, text):
    ax.set_title(f'({letter})  {text}', loc='left', fontsize=9,
                 fontweight='bold', pad=7)
