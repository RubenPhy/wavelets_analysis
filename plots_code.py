import os
import matplotlib.pyplot as plt

def plot_two_axis(
    primary_series,
    secondary_series,
    *,
    primary_label="Serie primaria",
    secondary_label="Serie secundaria",
    primary_ylabel=None,
    secondary_ylabel=None,
    title=None,
    output_dir="plots",
    filename="plot.png",
    figsize=(12, 6),
    secondary_color="red",
    show=True
):
    """
    Dibuja dos series (mismo índice) con ejes y independientes.

    Parameters
    ----------
    primary_series : pandas.Series
        Serie que se muestra en el eje y primario.
    secondary_series : pandas.Series
        Serie que se muestra en el eje y secundario.
    primary_label : str, default "Serie primaria"
        Etiqueta de la serie primaria en la leyenda.
    secondary_label : str, default "Serie secundaria"
        Etiqueta de la serie secundaria en la leyenda.
    primary_ylabel : str or None
        Texto para el eje y primario.  Si es None -> usa primary_label.
    secondary_ylabel : str or None
        Texto para el eje y secundario. Si es None -> usa secondary_label.
    title : str or None
        Título del gráfico. Si es None -> se deja sin título.
    output_dir : str, default "plots"
        Carpeta donde se guarda la imagen PNG (se crea si no existe).
    filename : str, default "plot.png"
        Nombre del archivo dentro de output_dir.
    figsize : tuple, default (12, 6)
        Tamaño de la figura.
    secondary_color : str, default "red"
        Color de la serie secundaria.
    show : bool, default True
        Si True, ejecuta plt.show(); si False, cierra la figura tras guardarla.

    Returns
    -------
    str
        Ruta completa del archivo guardado.
    """
    # ── Validaciones básicas ────────────────────────────────────
    if not primary_series.index.equals(secondary_series.index):
        raise ValueError("Las dos series deben tener exactamente el mismo índice.")

    # ── Preparar destino ───────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # ── Crear figura y ejes ────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize)

    # Eje y primario
    ax1.plot(
        primary_series.index,
        primary_series,
        label=primary_label,
        alpha=0.5
    )
    ax1.set_ylabel(primary_ylabel or primary_label)

    # Eje y secundario
    ax2 = ax1.twinx()
    ax2.plot(
        secondary_series.index,
        secondary_series,
        label=secondary_label,
        color=secondary_color
    )
    ax2.set_ylabel(secondary_ylabel or secondary_label)

    # Leyenda combinada (handles + labels de ambos ejes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Título y layout
    if title:
        plt.title(title)
    plt.tight_layout()

    # Guardar y mostrar
    plt.savefig(file_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return file_path
