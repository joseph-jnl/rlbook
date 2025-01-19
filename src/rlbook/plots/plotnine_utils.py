from io import BytesIO
from plotnine import ggplot
from matplotlib.pyplot import imread, subplots


def subplot(*args: ggplot, rows: int = 1, cols: int = 2, figsize:tuple[int,int]=(15, 8)):
    """Subplot multiple plotnine figures
    Args:
        *args: plotnine ggplot figure objects
        rows: number of rows
        cols: number of columns
        figsize: figure size
    """
    buffers = []
    for i, f in enumerate(args):
        buffers.append(BytesIO())
        f.save(buffers[i], format="png", dpi=300, verbose=False)
        buffers[i].seek(0)

    fig, axes = subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        img = imread(buffers[i], format="png")
        ax.imshow(img)
        ax.set_axis_off()

    fig.tight_layout()

    return fig
