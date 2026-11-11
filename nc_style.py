"""Shared figure style for the Nature Cities submission.
Canvas drawn at final size (full width 180 mm, single column 88 mm, height <= 170 mm), Arial, labels 8 pt, ticks and legends 7 pt,
panel letters 10 pt bold lowercase, lines 0.5-1 pt, markers 3-4 pt, vector PDF with editable text (fonttype 42),
map fills rasterised at 600 dpi. One colour per source in every figure."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
MM=1/25.4
FULL=180*MM; SINGLE=88*MM; MAXH=170*MM
plt.rcParams.update({
    "font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
    "legend.title_fontsize":7,"axes.linewidth":0.5,"lines.linewidth":0.8,"lines.markersize":3.5,"patch.linewidth":0.5,
    "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.minor.width":0.4,"ytick.minor.width":0.4,
    "xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.minor.size":1.5,"ytick.minor.size":1.5,"xtick.major.pad":1.5,"ytick.major.pad":1.5,
    "axes.labelpad":2,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"legend.handlelength":1.2,
    "legend.handletextpad":0.4,"legend.borderaxespad":0.2,"legend.labelspacing":0.3,"grid.linewidth":0.4,"grid.color":"#E6E6E6",
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":600,"figure.dpi":150,"hatch.linewidth":0.5,"mathtext.default":"regular"})
# colours: one per source, shared by all figures
SURGE10,SURGE100,SURGE1000="#A0522D","#E39A6A","#F3D9C4"     # surge: browns/oranges, darker = more frequent
SURGE=SURGE100
RAIN_MOD,RAIN_EXT="#08306B","#4292C6"; RAIN=RAIN_EXT            # rain: blues
TIDE="#2E7D5B"                                                  # tide: green
PROT="#4D4D4D"; PROT_LIGHT="#A6A6A6"                            # protection instruments / insurance: dark grey
NEUTRAL="#BDBDBD"; LAND="#FCFCF9"; WATER="#E4E4E4"; INK="#1A1A1A"; SANDY="#8E1B1B"
def panel(ax,letter,dx=-7,dy=2,fig=None):
    """bold 10 pt lowercase letter, placed dx/dy mm from the axes' top-left corner."""
    f=fig or ax.figure
    t=ax.transAxes+ScaledTranslation(dx*MM,dy*MM,f.dpi_scale_trans)
    ax.text(0,1,letter,transform=t,fontsize=10,fontweight="bold",va="bottom",ha="left")
def axes_mm(fig,x,y,w,h):
    """add axes by position in mm from the figure's bottom-left corner."""
    W,H=fig.get_size_inches()/MM
    return fig.add_axes([x/W,y/H,w/W,h/H])
def save(fig,name,outdirs=("figures",)):
    import os
    if os.environ.get("FIG_OUTDIR"): outdirs=tuple(os.environ["FIG_OUTDIR"].split(","))   # preview renders go elsewhere
    for d in outdirs:
        os.makedirs(d,exist_ok=True)
        fig.savefig(f"{d}/{name}.pdf"); fig.savefig(f"{d}/{name}.png",dpi=300)
    print("saved",name,"| size mm: %.0f x %.0f"%tuple(fig.get_size_inches()/MM))
