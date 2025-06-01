import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
import os

def plot_two_axis(
    primary_series,
    secondary_series,
    *,
    primary_label="Primary series",
    secondary_label="Secondary series",
    primary_ylabel=None,
    secondary_ylabel=None,
    title=None,
    output_dir="plots",
    filename="plot.png",
    figsize=(12, 6),
    secondary_color=None,
    secondary_alpha=0.65,
    primary_alpha=0.50,
    show=True
):
    """
    Plot two time-aligned series with independent y-axes without letting the
    secondary line dominate the primary one.

    Parameters
    ----------
    primary_series : pandas.Series
    secondary_series : pandas.Series
    primary_label, secondary_label : str
        Legend labels for each series.
    primary_ylabel, secondary_ylabel : str or None
        Axis labels (falls back to *label* if None).
    secondary_color : str or None, default None
        If None, picks a muted colour from Seaborn (softer than raw “red/green”).
    secondary_alpha : float, default 0.65
        Transparency of the secondary line (0 = invisible, 1 = opaque).
    primary_alpha : float, default 0.50
        Transparency of the primary line.
    title : str or None
        Figure title.
    output_dir : str, default "plots"
        Folder where the PNG is saved.
    filename : str, default "plot.png"
        File name inside *output_dir*.
    figsize : tuple, default (12, 6)
        Figure size in inches.
    show : bool, default True
        Whether to display the plot (True) or just save it (False).

    Returns
    -------
    str
        Full path of the saved PNG.
    """
    # 1 · Validación básica
    if not primary_series.index.equals(secondary_series.index):
        raise ValueError("Both series must share the same DatetimeIndex.")

    # 2 · Elegir color pastel si el usuario no fijó uno
    if secondary_color is None:
        secondary_color = sns.color_palette("muted")[2]   # azul verdoso suave

    # 3 · Preparar destino
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # 4 · Estilo general Seaborn
    sns.set_theme(
        rc={
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )

    # 5 · Crear figura + ejes
    fig, ax1 = plt.subplots(figsize=figsize)

    #   · Eje primario
    ax1.plot(
        primary_series.index,
        primary_series,
        label=primary_label,
        alpha=primary_alpha
    )
    ax1.set_ylabel(primary_ylabel or primary_label, fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)

    #   · Eje secundario
    ax2 = ax1.twinx()
    ax2.plot(
        secondary_series.index,
        secondary_series,
        label=secondary_label,
        color=secondary_color,
        alpha=secondary_alpha
    )
    ax2.set_ylabel(secondary_ylabel or secondary_label, fontsize=16)
    ax2.tick_params(axis='y', labelsize=11)

    # 6 · Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=16)

    # 7 · Título y guardado
    if title:
        plt.title(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(file_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return file_path
