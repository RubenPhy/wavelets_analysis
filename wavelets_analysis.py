import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D   # para la leyenda “manual”
from plots_code import plot_two_axis

# —-—-—- Ajustes globales Seaborn —-—-—-
sns.set_theme(style="darkgrid")          # fondo gris con rejilla en gris oscuro
plt.rcParams.update({"axes.titleweight": "bold"})  # (opcional) negrita en títulos

# --- XXX Comentarios Generales ---
# 1) Ejes y leyendas: Se ha intentado que sean descriptivos.
# 2) Sentido de los gráficos y nombres: Revisados para mayor claridad.
# 3) Plot de Señal Denoised H_i: Añadido.
# ---------------------------------

# Paso 1: Cargar y preparar los datos
ticker_name = 'SP' #'AAP.N'
# path = f'raw/{ticker_name}-2012-03-29-2024-03-01.csv'
# df = pd.read_csv(path, index_col='Date', parse_dates=True)
# df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
# df_log_ret = df['log_return'].dropna() 
# initial_price = df['CLOSE'].dropna().iloc[0]

path = 'sector_log_returns.csv'
df = pd.read_csv(path, index_col='Date', parse_dates=True)
df_log_ret = df['S&P 500'].dropna()
initial_price = 1

# Calcular precios acumulados para visualización
cumulative_log_returns = df_log_ret.cumsum()
cumulative_prices = initial_price * np.exp(cumulative_log_returns)
cumulative_prices = pd.Series(cumulative_prices, index=df_log_ret.index, name='Cumulative_Price')

# Ajustar la longitud a una potencia de 2 para simplificar el análisis wavelet
n = len(df_log_ret)
n_adjusted = 2**int(np.log2(n))
df_log_ret = df_log_ret.iloc[:n_adjusted]
cumulative_prices = cumulative_prices.iloc[:n_adjusted]
print(f"Longitud de la serie ajustada: {len(df_log_ret)}")

# Paso 2: Descomposición con Haar Wavelets hasta nivel 4
level = 4
coeffs = pywt.wavedec(df_log_ret.values, 'haar', level=level, mode='periodization') # coeffs = [cA_L, cD_L, ..., cD_1]
print("Longitudes de los coeficientes:")
# Los coeficientes son [cA_level, cD_level, cD_{level-1}, ..., cD_1]
# Por lo tanto, coeffs[0] es cA4, coeffs[1] es cD4, coeffs[2] es cD3, coeffs[3] es cD2, coeffs[4] es cD1.
print(f"cA{level}: {len(coeffs[0])}")
for i_coeff in range(1, len(coeffs)):
    print(f"cD{level - i_coeff + 1}: {len(coeffs[i_coeff])}")

# Visualizar coeficientes wavelet
fig, axs = plt.subplots(len(coeffs), 1, figsize=(12, 10), sharex=False)
# ­­­­-------------------------- cA (aproximación) ---------------------------­
axs[0].plot(coeffs[0])
axs[0].set_title(f'Approximation Coefficients (cA{level})', fontsize=18)
axs[0].tick_params(axis='both', labelsize=14)
# ­­­­-------------------------- cD (detalle) ---------------------------­
for i_coeff in range(1, len(coeffs)):
    axs[i_coeff].plot(coeffs[i_coeff])
    axs[i_coeff].set_title(f'Detail Coefficients (cD{level - i_coeff + 1})', fontsize=18)
    axs[i_coeff].tick_params(axis='both', labelsize=14)
# ­­­­-------------------------- Título general y exportación ------------------------­
fig.suptitle('Wavelet Coefficients of Log Returns', fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.96])             # deja sitio al título general
# Guardar y mostrar
plt.savefig(f'plots/wavelet_coefficients_log_returns_{ticker_name}.png')
print(f"Wavelet coefficient plot saved to: plots/wavelet_coefficients_log_returns_{ticker_name}.png")
plt.show()
plt.close()

# Visualizar 
top_pct = 3
# ---------- Función para localizar fechas extremas ----------
def extract_extreme_dates(coeffs, level, base_index, top_pct=5):
    """
    Devuelve un dict {coef_name: DatetimeIndex} con las fechas cuyas
    oscilaciones (|coef|) están en el top_pct % más alto.
    
    base_index es el índice temporal de la serie original de log-returns.
    El mapeo coef → fecha es aproximado: j-ésimo coeficiente a nivel k
    se asocia al momento base_index[j * 2**k].
    """
    extreme_dates = {}
    for i, arr in enumerate(coeffs):
        if i == 0:                    # Aproximación
            name   = f'cA{level}'
            scale  = 2**level
        else:                         # Detalle
            k      = level - i + 1
            name   = f'cD{k}'
            scale  = 2**k
        # Umbral p-ésimo
        thr = np.percentile(np.abs(arr), 100 - top_pct)
        idx = np.where(np.abs(arr) >= thr)[0]
        # Mapear a fechas (asegura no salir del rango)
        pos = np.minimum(idx * scale, len(base_index) - 1)
        extreme_dates[name] = base_index[pos]
    return extreme_dates

# Fechas extremas para todos los coeficientes
extreme_dates = extract_extreme_dates(coeffs, level, df_log_ret.index, top_pct)

# ---------- Plot de precios + líneas verticales ----------
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(cumulative_prices, linewidth=1.2) # o df['CLOSE']
ax.set_title(f'Cumulative Price with Top {top_pct}% Wavelet Oscillations', fontsize=22)
ax.set_ylabel('Price', fontsize=18)
ax.set_xlabel('Date', fontsize=18)

# Increase tick label size
ax.tick_params(axis='both', labelsize=15)

# Colores distintos para cada conjunto de líneas
palette = sns.color_palette('husl', n_colors=len(extreme_dates))

for (coef_name, dates), color in zip(extreme_dates.items(), palette):
    for date in dates:
        ax.axvline(date, color=color, linewidth=1.0, linestyle='--', alpha=0.7)  # <-- Más ancho

# Leyenda manual (una entrada por coeficiente)
handles = [Line2D([0], [0], color=c, lw=3, linestyle='--', label=n)
           for (n, _), c in zip(extreme_dates.items(), palette)]
ax.legend(handles=handles, title='Coefficient band', loc='upper left', fontsize=14, title_fontsize=15)
plt.tight_layout()
plt.savefig(f'plots/dates_with_highest_coeff_{ticker_name}.png')
plt.show()
plt.close()

# Paso 3: Calcular H_i(t) para i=1,...,level
sns.set_theme(
    rc={
        "axes.titlesize": 22,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

# Initialize dictionaries for H_i and detail reconstructions
H_series = {}
detail_reconstructed_series = {}  # Store detail_reconstructed for each level
print("\n--- Starting H_i(t) computation and detail reconstruction ---")
for i_level in range(1, level + 1):  # i_level is the i in H_i
    abs_detail_sum = np.zeros_like(df_log_ret.values, dtype=float)
    detail_reconstructed_series[i_level] = {}
    print(f"\nComputing H_{i_level} …")

    # ------- Reconstruct each detail D_j and accumulate its |·| -------
    for j_level in range(1, i_level + 1):  # j_level is the j in D_j
        idx_coeff = level - j_level + 1  # Position of cD_j in coeffs

        # Build a coefficient list with only cD_j kept, others → 0
        rec_coeff_list = [np.zeros_like(coeffs[0])]  # Null cA_L
        for idx in range(1, len(coeffs)):
            rec_coeff_list.append(coeffs[idx] if idx == idx_coeff else np.zeros_like(coeffs[idx]))

        # Reconstruct time-domain detail D_j(t)
        try:
            detail_rec = pywt.waverec(rec_coeff_list, 'haar', mode='periodization')
            detail_rec = detail_rec[:len(abs_detail_sum)]
        except ValueError as err:
            raise RuntimeError(f"pywt.waverec failed for D_{j_level}: {err}")

        abs_detail_sum += np.abs(detail_rec)
        detail_reconstructed_series[i_level][j_level] = detail_rec

        # ── Plot cumulative price vs. reconstructed detail D_j ──
        detail_rec_series = pd.Series(detail_rec, index=cumulative_prices.index,
                                      name=f"Detail_D{j_level}")
        file_path = plot_two_axis(
            primary_series=cumulative_prices,
            secondary_series=detail_rec_series,
            primary_label="Cumulative Price",
            secondary_label=f"Detail D{j_level} (from log returns)",
            primary_ylabel="Cumulative Price",
            secondary_ylabel=f"Amplitude of Detail D{j_level}",
            title=f"Cumulative Price vs. Reconstructed Detail D{j_level} for H{i_level}",
            filename=f"detail_D{j_level}_reconstructed_for_H{i_level}_{ticker_name}.png",
            secondary_color="red",
            show=False
        )
        print(f"  Plot saved to: {file_path}")

    # ── Plot cumulative price vs. Σ|D_j| (up to j = i_level) ──
    abs_detail_sum_series = pd.Series(abs_detail_sum, index=cumulative_prices.index,
                                      name=f"AbsDetailSum_H{i_level}")
    file_path = plot_two_axis(
        primary_series=cumulative_prices,
        secondary_series=abs_detail_sum_series,
        primary_label="Cumulative Price",
        secondary_label=f"Sum |D_j| up to j={i_level} (for H_{i_level})",
        primary_ylabel="Cumulative Price",
        secondary_ylabel="Absolute sum of details",
        title=f"Cumulative Price vs. Absolute Detail Sum for H_{i_level}",
        filename=f"sum_abs_details_for_H{i_level}_{ticker_name}.png",
        secondary_color="green",
        show=False
    )
    print(f"  Plot saved to: {file_path}")

    # ------- Construct approximation A_L (all details → 0) -------
    approx_coeff_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx = pywt.waverec(approx_coeff_list, 'haar', mode='periodization')
    approx = approx[:len(abs_detail_sum)]
    approx = np.where(approx == 0, 1e-10, approx)  # Avoid division by zero

    # ------- Compute H_i(t) = Σ|D_j| / A_L -------
    H_series[i_level] = abs_detail_sum / approx
    print(f"  H_{i_level} computed, length: {len(H_series[i_level])}")

# ------- Create subplots for each H_i with D_1 to D_j and H_i stacked -------
for i_level in range(1, level + 1):
    # Number of subplots: i_level details + H_i
    n_subplots = i_level + 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 4 * n_subplots), sharex=True)
    
    # Ensure axes is a list even for a single subplot
    if n_subplots == 1:
        axes = [axes]
    
    # Plot each detail D_j for j=1 to i_level with cumulative price
    for j_level in range(1, i_level + 1):
        ax1 = axes[j_level - 1]
        ax2 = ax1.twinx()  # Secondary y-axis for cumulative price
        detail_series = pd.Series(
            detail_reconstructed_series[i_level][j_level],
            index=df_log_ret.index,
            name=f"Detail_D{j_level}"
        )
        # Plot detail D_j
        sns.lineplot(
            data=detail_series,
            ax=ax1,
            color="green",
            label=f"Detail D{j_level}",
            alpha=0.65
        )
        # Plot cumulative price
        ax2.plot(
            cumulative_prices.index,
            cumulative_prices,
            label="Cumulative Price"
        )
        ax1.set_ylabel(f"D{j_level} Amplitude", fontsize=14)
        ax2.set_ylabel("Cumulative Price", fontsize=14)
        ax1.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.2f'))  # 2 decimals for D_j
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    # Plot H_i in the last subplot with cumulative price
    ax1 = axes[-1]
    ax2 = ax1.twinx()
    h_series = pd.Series(H_series[i_level], index=df_log_ret.index, name=f"H_{i_level}")
    sns.lineplot(
        data=h_series,
        ax=ax1,
        color="red",
        label=f"H_{i_level}(t)",
        alpha=0.65
    )
    ax2.plot(
        cumulative_prices.index,
        cumulative_prices,
        label="Cumulative Price",
    )
    ax1.set_ylabel(f"H_{i_level} Value (%)", fontsize=14)
    ax2.set_ylabel("Cumulative Price", fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    #ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))  # No decimals, percentage
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    # Set common x-axis label and title
    axes[-1].set_xlabel("Date", fontsize=14)
    axes[-1].tick_params(axis='x', labelsize=12)
    fig.suptitle(f"Reconstructed Details and H_{i_level} with Cumulative Price for {ticker_name}", fontsize=18)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save the subplot
    file_path = f'plots/H_{i_level}_with_details_and_price_{ticker_name}.png'
    plt.savefig(file_path)
    print(f"Subplot for H_{i_level} and details with cumulative price saved to: {file_path}")
    plt.show()

# ---------- Plot all H_i(t) curves together ----------
plt.figure(figsize=(14, 7))
# Plot all H_i(t) curves
for i_level in range(1, level + 1):
    h_series = pd.Series(H_series[i_level], index=df_log_ret.index, name=f"H_{i_level}")
    plt.plot(h_series.index, h_series.values, label=f'$H_{i_level}(t)$')

# Plot cumulative returns (cumulative_prices) on a secondary y-axis
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return')
ax2.set_ylabel('Cumulative Return', fontsize=20, color='black')
ax2.tick_params(axis='y', labelsize=16, colors='black')

# Main axis labels and legend
ax1.set_title(f'$H_i(t)$ across detail aggregation levels for {ticker_name}', fontsize=28)
ax1.set_xlabel('Date', fontsize=24)
ax1.set_ylabel('$H_i(t)$ (%)', fontsize=24)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, title='Aggregation level', loc='upper left', fontsize=16)

plt.tight_layout()
plt.savefig(f'plots/H_series_all_levels_{ticker_name}.png')
print(f"Plot of all H_series saved to: plots/H_series_all_levels_{ticker_name}.png")
plt.show()

# Paso 4: Calcular energía móvil (ventana de 30 días)
print("\n--- Iniciando Cálculo de Energía Móvil ---")
window_size = 30
energy_series = {}
for i_h_level in range(1, level + 1):
    # Asegurarse que H_series[i_h_level] es un array 1D para pd.Series
    h_series_array = np.asarray(H_series[i_h_level])
    energy_series[i_h_level] = pd.Series(h_series_array, index=df_log_ret.index).rolling(
        window=window_size, min_periods=1
    ).apply(lambda x: np.sum(x**2), raw=True).values
    print(f"Energía móvil H_{i_h_level}, longitud: {len(energy_series[i_h_level])}")

# Plot Energy_series todas juntas + cummulative returns
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot energy series
for i_h_level_plot in range(1, level + 1):
    ax1.plot(df_log_ret.index, energy_series[i_h_level_plot], label=f'$Energy(H_{i_h_level_plot})$')

ax1.set_xlabel('Fecha', fontsize=18)
ax1.set_ylabel('Energía Móvil de $H_i(t)$', fontsize=18)
ax1.tick_params(axis='both', labelsize=15)
ax1.set_title('Energía Móvil de $H_i(t)$ para cada Nivel (Ventana 30 días)', fontsize=22)

# Plot cumulative returns on secondary axis
ax2 = ax1.twinx()
ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return')
ax2.set_ylabel('Cumulative Return', fontsize=18)
ax2.tick_params(axis='y', labelsize=15)
ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.2f'))

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15, title_fontsize=16)

plt.tight_layout()
plt.savefig(f'plots/Energy_series_all_levels_{ticker_name}.png')
print(f"Gráfico de todas las Energy_series guardado en: plots/Energy_series_all_levels_{ticker_name}.png")
plt.show()

# Paso 5: Reducción de ruido (threshold_energy_retain % de energía retenida)
print("\n--- Iniciando Reducción de Ruido (Denoising) ---")
threshold_energy_retain = 0.4 # Se mantiene el 40% de la energía, como en el ejemplo. Modificado de 0.2 a 0.4
denoised_series = {}
for i_h_level in range(1, level + 1):
    coeffs_H = pywt.wavedec(H_series[i_h_level], 'haar', level=level, mode='periodization')
    flat_coeffs = np.concatenate([c for c in coeffs_H if c is not None])
    # Convierte la lista de arrays de coeficientes wavelet (coeffs_H) en un único array NumPy 1D (plano).
    # Esto facilita el cálculo de estadísticas globales sobre todos los coeficientes.
    # La cláusula 'if c is not None' es una medida de seguridad, aunque pywt.wavedec típicamente no devuelve elementos None.

    sorted_abs_coeffs = np.sort(np.abs(flat_coeffs))[::-1]
    # La magnitud de un coeficiente wavelet a menudo se correlaciona con su importancia.
    # Invierte el array ordenado, resultando en coeficientes ordenados por su
    #    valor absoluto de mayor a menor. Los coeficientes más "energéticos" o importantes quedan al principio.

    total_energy = np.sum(sorted_abs_coeffs**2)
    # Calcula la energía total contenida en la señal H_series[i] a través de sus coeficientes wavelet.
    # La energía de un coeficiente wavelet es proporcional a su cuadrado.
    # Esta es la suma de las energías de todos los coeficientes.
    
    if total_energy == 0: # Evitar error si la energía es cero
        print(f"  Advertencia: Energía total de H_{i_h_level} es cero. Denoised H_{i_h_level} será cero.")
        T = 0
    else:
        cumulative_energy = np.cumsum(sorted_abs_coeffs**2)
    # Calcula la suma acumulada de las energías de los coeficientes (que están ordenados de mayor a menor).
    # El primer elemento es la energía del coeficiente más grande.
    # El segundo elemento es la suma de las energías de los dos coeficientes más grandes, y así sucesivamente.
    # El último elemento de cumulative_energy es igual a total_energy.

        idx_threshold = np.searchsorted(cumulative_energy, threshold_energy_retain * total_energy)
    # 1. 0.4 * total_energy: Calcula el 40% de la energía total. Este es el objetivo de energía a retener.
    # 2. np.searchsorted(cumulative_energy, ...): Encuentra el índice 'j' en el array 'cumulative_energy'
    #    tal que todos los elementos hasta cumulative_energy[j-1] suman menos del 40% de la energía total,
    #    y al incluir cumulative_energy[j], se alcanza o supera ese 40%.
    #    En otras palabras, los 'j' coeficientes más grandes (según sorted_abs_coeffs) contienen
    #    aproximadamente el 40% de la energía total de la señal H_series[i].

        T = sorted_abs_coeffs[idx_threshold] if idx_threshold < len(sorted_abs_coeffs) else sorted_abs_coeffs[-1]
    # Establece el valor del umbral 'T'.
    # T es el valor absoluto del j-ésimo coeficiente más grande (es decir, el coeficiente más pequeño
    # entre los que contribuyen al 40% de la energía retenida).
    # Los coeficientes cuya magnitud sea menor que T serán considerados "ruido".
    # La condición 'if j < len(sorted_abs_coeffs)' maneja el caso límite donde 'j' podría
    # estar fuera de los límites del array (por ejemplo, si se retiene el 100% de la energía).

    # Umbral duro
    thresholded_coeffs = [pywt.threshold(c, T, mode='hard') for c in coeffs_H] # Asumiendo c nunca es None de wavedec
    # Aplica la técnica de "umbral duro" a cada conjunto de coeficientes en la lista original coeffs_H
    # (que eran [cA_H, cD_L_H, ..., cD_1_H]).
    # Para cada coeficiente en cada array 'c' de coeffs_H:
    #  - Si la magnitud del coeficiente (abs(coef)) es mayor que T, el coeficiente se mantiene sin cambios.
    #  - Si la magnitud del coeficiente (abs(coef)) es menor o igual a T, el coeficiente se establece en 0.
    # Esto elimina los coeficientes considerados "ruido".
    reconstructed_signal = pywt.waverec(thresholded_coeffs, 'haar', mode='periodization')
    denoised_series[i_h_level] = reconstructed_signal[:len(H_series[i_h_level])]
    # Reconstruye la señal H_series[i] utilizando los coeficientes wavelet umbralizados (thresholded_coeffs).
    # El resultado, denoised_series[i], es una versión con ruido reducido (denoised) de la H_series[i] original.
    # Se espera que esta serie "limpia" revele de forma más clara los picos o características significativas.
    # [:len(H_series[i])] asegura que la serie reconstruida tenga la misma longitud que la H_series[i] original.

    print(f"Señal Denoised H_{i_h_level} calculada, longitud: {len(denoised_series[i_h_level])}. Umbral T={T:.4f}")

# --- Plot de H_i observada vs. Denoised H_i (similar a p.23 del PDF) ---
level_to_plot_denoising = 1 # Elige un nivel, por ejemplo H_1
if level_to_plot_denoising in H_series and level_to_plot_denoising in denoised_series:
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df_log_ret.index, H_series[level_to_plot_denoising], label=f'$H_{level_to_plot_denoising}$ (Observada)')
    plt.title(f'$H_{level_to_plot_denoising}(t)$ (Observada)')
    plt.ylabel('Amplitud')
    plt.legend()

    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.plot(df_log_ret.index, denoised_series[level_to_plot_denoising], label=f'Denoised $H_{level_to_plot_denoising}$', color='orange')
    ax1.set_ylabel('Amplitud')
    ax2 = ax1.twinx()
    ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return', alpha=0.5)
    ax2.set_ylabel('Cumulative Return')
    ax1.set_xlabel('Fecha')
    plt.title(f'Denoised $H_{level_to_plot_denoising}(t)$ y Cumulative Return ({threshold_energy_retain*100}% Energía Retenida)')
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/H{level_to_plot_denoising}_vs_denoised_{ticker_name}.png')
    print(f"\nGráfico de H_{level_to_plot_denoising} y su versión denoised guardado en: plots/H{level_to_plot_denoising}_vs_denoised_{ticker_name}.png")
    plt.show()

# --- Plot de todas las Denoised H_series (similar a p.24 del PDF) ---
plt.figure(figsize=(16, 10))
ax1 = plt.gca()
# Plot all denoised H_i(t) series
for i_plot in range(1, level + 1):
    if i_plot in denoised_series:
        ax1.plot(df_log_ret.index[:len(denoised_series[i_plot])], denoised_series[i_plot], label=f'Denoised $H_{i_plot}$')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Amplitud Denoised $H_i(t)$')
ax1.set_title(f'Denoised $H_i(t)$ para cada Nivel ({threshold_energy_retain*100}% Energía Retenida)')
ax1.tick_params(axis='y', labelsize=12)

# Plot cumulative returns on secondary axis
ax2 = ax1.twinx()
ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return', color='black', alpha=0.5)
ax2.set_ylabel('Cumulative Return')
ax2.tick_params(axis='y', labelsize=12)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(f'plots/denoised_H_series_all_levels_{ticker_name}.png')
print(f"Gráfico de todas las Denoised H_series guardado en: plots/denoised_H_series_all_levels_{ticker_name}.png")
plt.show()


# Paso 6: Identificar fechas críticas y Sistema de contador para señales de alerta
print("\n--- Iniciando Identificación de Fechas Críticas y Sistema de Alerta ---")
critical_dates = {}
for i_h_level in range(1, level + 1):
    critical_indices = np.where(np.abs(denoised_series[i_h_level]) > 1e-10)[0] # Tolerancia para ceros
    critical_dates[i_h_level] = df_log_ret.index[critical_indices]
    print(f"H_{i_h_level} - Fechas críticas iniciales identificadas: {len(critical_dates[i_h_level])}")

# --- Modificación del Paso 6 para almacenar contadores ---
all_critical_date_counters = {i: {} for i in range(1, level + 1)} # Para almacenar crit_date: counter
early_warnings = {} # Diccionario para almacenar las fechas de alerta temprana finales

for i_h_level in range(1, level + 1):
    early_warnings[i_h_level] = []
    print(f"\nProcesando sistema de contador para H_{i_h_level}:")
    if not critical_dates[i_h_level].empty: # Solo procesar si hay fechas críticas
        for crit_date_idx, crit_date in enumerate(critical_dates[i_h_level]):
            # print(f"  Verificando robustez de fecha crítica {crit_date_idx+1}/{len(critical_dates[i_h_level])}: {crit_date.date()}")
            original_series_idx = df_log_ret.index.get_loc(crit_date)
            counter = 0 # Inicializa un contador para esta fecha crítica específica.

            # Este bucle simula "terminar" la serie de precios en diferentes puntos 'x'
            # alrededor de la fecha crítica 'crit_idx'.
            # Se define una ventana de 31 días: 15 días antes de crit_idx, crit_idx mismo, y 15 días después.
            # max(0, crit_idx - 15): asegura que no empecemos antes del inicio de la serie.
            # min(len(df_log_ret), crit_idx + 16): asegura que no nos pasemos del final de la serie. 
            # Bucle de simulación para robustez
            for x_sim_idx in range(max(0, original_series_idx - 15), min(len(df_log_ret), original_series_idx + 16)):
                truncated = df_log_ret.iloc[:x_sim_idx+1].values
                if not truncated.size: continue # Saltar si truncado está vacío
                # -- Inicio de la Simulación para el punto 'x' --
                # El objetivo es ver si la fecha 'crit_date' todavía se detectaría como
                # anómala si solo hubiéramos tenido datos hasta el día 'x'.

                # Crear serie modificada
                # truncated: Toma la serie de retornos logarítmicos ('df_log_ret') desde el inicio hasta el día 'x'.                
                flipped = truncated[::-1]
                # modified: Crea una nueva serie de longitud igual a la 'df_log_ret' original.
                # Lo hace concatenando la 'truncated' y su 'flipped' repetidamente.
                # Esto es una técnica para extender artificialmente la serie truncada para el análisis wavelet,
                # tratando de minimizar los efectos de borde y mantener cierta estructura estadística.
                # (len(df_log_ret) // (2 * (x+1)) + 1): Calcula cuántas veces se necesita repetir el patrón [truncated, flipped].

                # Asegurar que el patrón para concatenar no sea vacío
                pattern_len = 2 * len(truncated)
                if pattern_len == 0: continue

                num_repeats = (len(df_log_ret) // pattern_len) + 1
                modified = np.concatenate([truncated, flipped] * num_repeats)[:len(df_log_ret)]
                
                coeffs_mod = pywt.wavedec(modified, 'haar', level=level, mode='periodization')
                details_sum_mod = np.zeros_like(df_log_ret.values, dtype=float)
                
                for j_detail_level in range(1, i_h_level + 1):
                    idx_Dj_in_coeffs_mod = level - j_detail_level + 1
                    coeff_list_detail_mod_reconstruct = [np.zeros_like(coeffs_mod[0])]
                    for k_idx_mod in range(1, len(coeffs_mod)):
                        if k_idx_mod == idx_Dj_in_coeffs_mod:
                            coeff_list_detail_mod_reconstruct.append(coeffs_mod[k_idx_mod])
                        else:
                            coeff_list_detail_mod_reconstruct.append(np.zeros_like(coeffs_mod[k_idx_mod]))
                    
                    detail_reconstructed_mod_val = pywt.waverec(coeff_list_detail_mod_reconstruct, 'haar', mode='periodization')
                    details_sum_mod += np.abs(detail_reconstructed_mod_val[:len(details_sum_mod)])

                coeff_list_approx_mod_reconstruct = [coeffs_mod[0]]
                for k_idx_mod_detail in range(1, len(coeffs_mod)):
                    coeff_list_approx_mod_reconstruct.append(np.zeros_like(coeffs_mod[k_idx_mod_detail]))
                
                approx_mod = pywt.waverec(coeff_list_approx_mod_reconstruct, 'haar', mode='periodization')[:len(details_sum_mod)]
                approx_mod = np.where(approx_mod == 0, 1e-10, approx_mod)
                H_mod = details_sum_mod / approx_mod
                
                # Denoising de H_mod
                coeffs_H_mod = pywt.wavedec(H_mod, 'haar', level=level, mode='periodization')
                flat_coeffs_mod = np.concatenate([c for c in coeffs_H_mod if c is not None])
                sorted_abs_coeffs_mod = np.sort(np.abs(flat_coeffs_mod))[::-1]
                
                # Importante: 'total_energy' aquí se refiere a la energía de H_series[i_h_level] ORIGINAL,
                # calculada en el Paso 5. El paper (p.25) es un poco ambiguo si el umbral T
                # debe recalcularse basado en la energía de H_mod o si se usa el T original.
                # Esta implementación usa la energía de H_series[i_h_level] original para determinar T_mod,
                # que es lo que hacía el código previo.
                # Si se quisiera usar la energía de H_mod:
                # total_energy_mod_val = np.sum(sorted_abs_coeffs_mod**2)
                # if total_energy_mod_val == 0: T_mod_val = 0
                # else:
                #    cumulative_energy_mod_val = np.cumsum(sorted_abs_coeffs_mod**2)
                #    idx_threshold_mod = np.searchsorted(cumulative_energy_mod_val, threshold_energy_retain * total_energy_mod_val)
                #    T_mod_val = sorted_abs_coeffs_mod[idx_threshold_mod] if idx_threshold_mod < len(sorted_abs_coeffs_mod) else sorted_abs_coeffs_mod[-1]
                # Esta línea crítica usa 'total_energy' de la H_series original del Paso 5
                # Para la energía de H_mod (como podría interpretarse del paper):
                # total_energy_for_T_mod = np.sum(sorted_abs_coeffs_mod**2)

                # Por ahora, mantenemos la lógica original del código del usuario para 'total_energy' en este bloque
                # que se refiere a la 'total_energy' de la H_series[i_h_level] original del Paso 5.
                # Esta es la 'total_energy' de la H_series[i_h_level] NO MODIFICADA (del Paso 5)
                current_total_energy_for_thresholding = np.sum(np.sort(np.abs(np.concatenate([c for c in pywt.wavedec(H_series[i_h_level], 'haar', level=level, mode='periodization') if c is not None])))[::-1]**2)

                if current_total_energy_for_thresholding == 0:
                    T_mod_val = 0
                else:
                    cumulative_energy_mod = np.cumsum(sorted_abs_coeffs_mod**2)
                    # El umbral de energía se basa en el 40% de la energía total de la H_series[i] ORIGINAL
                    idx_threshold_mod = np.searchsorted(cumulative_energy_mod, threshold_energy_retain * current_total_energy_for_thresholding)
                    T_mod_val = sorted_abs_coeffs_mod[idx_threshold_mod] if idx_threshold_mod < len(sorted_abs_coeffs_mod) else sorted_abs_coeffs_mod[-1]

                thresholded_coeffs_mod_list = [pywt.threshold(c, T_mod_val, mode='hard') for c in coeffs_H_mod]
                denoised_mod = pywt.waverec(thresholded_coeffs_mod_list, 'haar', mode='periodization')[:len(H_mod)]

                if original_series_idx < len(denoised_mod) and np.abs(denoised_mod[original_series_idx]) > 1e-10:
                    counter += 1
            
            all_critical_date_counters[i_h_level][crit_date] = counter # Almacenar contador
            if counter > 10: # Umbral del paper p.25
                early_warnings[i_h_level].append(crit_date)
        print(f"  H_{i_h_level} - Señales de alerta temprana robustas calculadas: {len(early_warnings[i_h_level])}")
    else:
        print(f"  H_{i_h_level} - No hay fechas críticas iniciales para procesar.")


# --- Resultados del Sistema de Alerta ---
print("\n--- Resultados Finales del Sistema de Alerta Temprana ---")
for i_res in range(1, level + 1):
    print(f"H_{i_res} - Señales de alerta temprana detectadas:")
    if early_warnings[i_res]:
        for date_alert in early_warnings[i_res]:
            print(f"  {date_alert.strftime('%Y-%m-%d')}")
    else:
        print("  Ninguna señal de alerta temprana detectada.")

# --- Plot de Precios Acumulados con Señales de Alerta (ejemplo para H_4) ---
# Seleccionar el nivel de H_series para el cual mostrar las alertas
# Puedes cambiar esto o iterar si quieres mostrar alertas de múltiples niveles H_i

level_to_plot_warnings = level # Mostrar alertas del nivel más agregado por defecto
plt.figure(figsize=(15, 7))
plt.plot(cumulative_prices.index, cumulative_prices, label=f'Precios Acumulados ({path.split("/")[-1].split("-")[0]})', color='blue', alpha=0.7)

if level_to_plot_warnings in early_warnings and early_warnings[level_to_plot_warnings]:
    alert_dates_to_plot = early_warnings[level_to_plot_warnings]
    unique_label_alert = f'Alerta Temprana H_{level_to_plot_warnings}'
    for alert_idx, alert_date in enumerate(alert_dates_to_plot):
        plt.axvline(x=alert_date, color='red', linestyle='--', linewidth=1.2, label=unique_label_alert if alert_idx == 0 else None)
    print(f"\nMostrando alertas para H_{level_to_plot_warnings} en el gráfico de precios.")

plt.title(f'Precios Acumulados y Señales de Alerta Temprana para $H_{level_to_plot_warnings}$')
plt.xlabel('Fecha')
plt.ylabel('Precio Acumulado')
handles, labels = plt.gca().get_legend_handles_labels()
if handles: # Solo mostrar leyenda si hay algo que etiquetar
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.savefig(f'plots/cumulative_prices_with_H{level_to_plot_warnings}_warnings_{ticker_name}.png')
plt.show()
print(f"Gráfico de precios con alertas H_{level_to_plot_warnings} guardado en: plots/cumulative_prices_with_H{level_to_plot_warnings}_warnings_{ticker_name}.png")


# --- Plot de Frecuencia de Fechas Críticas Detectadas (similar a p.26 del PDF) ---
plt.figure(figsize=(16, 8))
plot_colors = ['blue', 'green', 'purple', 'orange'] # Colores para cada H_i

# Identificar fechas de crisis relevantes para el dataset específico si es posible
# Ejemplo con fechas genéricas del paper (ajustar si es necesario para AAP.N)
crisis_dates_in_data = {
    # "Dotcom": pd.to_datetime("2000-03-10"), # Probablemente no en el rango de AAP.N
    "Lehman": pd.to_datetime("2008-09-15"), # Probablemente no en el rango de AAP.N
    "COVID19": pd.to_datetime("2020-03-13")  # Podría estar en el rango
}
# Filtrar crisis que caen dentro del rango de fechas de 'df_log_ret'
valid_crisis_dates = {
    name: dt for name, dt in crisis_dates_in_data.items() 
    if df_log_ret.index.min() <= dt <= df_log_ret.index.max()
}

for i_scatter in range(1, level + 1):
    if i_scatter in all_critical_date_counters and all_critical_date_counters[i_scatter]:
        dates = list(all_critical_date_counters[i_scatter].keys())
        counts = list(all_critical_date_counters[i_scatter].values())
        
        is_alert_for_legend = False # Para la etiqueta de leyenda
        not_alert_for_legend = False

        for k_date_idx, date_val in enumerate(dates):
            is_current_alert = date_val in early_warnings[i_scatter]
            color_point = 'red' if is_current_alert else plot_colors[i_scatter-1]
            marker_point = 'o' if is_current_alert else 'x'
            size_point = 50 if is_current_alert else 30
            
            current_label = None
            if is_current_alert and not is_alert_for_legend:
                current_label = f'Alerta Temprana (H_{i_scatter}, Contador > 10)'
                is_alert_for_legend = True
            elif not is_current_alert and not not_alert_for_legend:
                current_label = f'Fecha Crítica (H_{i_scatter}, Contador <= 10)'
                not_alert_for_legend = True
                
            plt.scatter(date_val, counts[k_date_idx], color=color_point, marker=marker_point, s=size_point, label=current_label, alpha=0.7, edgecolors='k', linewidths=0.5)

plt.axhline(y=10, color='black', linestyle='--', linewidth=1.2, label='Umbral de Alerta (Contador = 10)')

for crisis_name, crisis_dt_val in valid_crisis_dates.items():
    plt.axvline(x=crisis_dt_val, color='dimgray', linestyle=':', linewidth=1.8, label=f'Evento Crisis: {crisis_name}')

plt.title('Validación de Fechas Críticas: Robustez y Señales de Alerta')
plt.xlabel('Fecha de la Fecha Crítica Original')
plt.ylabel('Contador de Robustez (Detecciones en Ventana Móvil de Simulación)')

handles_scatter, labels_scatter = plt.gca().get_legend_handles_labels()
if handles_scatter:
    by_label_scatter = dict(zip(labels_scatter, handles_scatter))
    plt.legend(by_label_scatter.values(), by_label_scatter.keys(), loc='upper left', bbox_to_anchor=(1,1))

plt.ylim(bottom=0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para leyenda fuera
plt.savefig(f'plots/frequency_critical_dates_robustness_check_{ticker_name}.png', bbox_inches='tight')
print(f"\nGráfico de frecuencia de robustez de fechas críticas guardado en: plots/frequency_critical_dates_robustness_check_{ticker_name}.png")
plt.show()

print("\n--- Análisis Completado ---")