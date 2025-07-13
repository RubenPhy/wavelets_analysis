import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Configuración global de Seaborn y Matplotlib
sns.set_theme(style="darkgrid")
plt.rcParams.update({"axes.titleweight": "bold"})

def load_and_prepare_data(file_path, column_name=None, initial_price=1):
    """
    Carga un archivo CSV y prepara los datos para el análisis.

    Parameters:
    - file_path (str): Ruta al archivo CSV.
    - column_name (str, optional): Nombre de la columna a usar para log returns. Si None, calcula desde 'CLOSE'.
    - initial_price (float): Precio inicial para calcular precios acumulados.

    Returns:
    - df_log_ret (pd.Series): Serie de log returns.
    - cumulative_prices (pd.Series): Precios acumulados.
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    if column_name:
        df_log_ret = df[column_name].dropna()
    else:
        df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
        df_log_ret = df['log_return'].dropna()
    
    cumulative_log_returns = df_log_ret.cumsum()
    cumulative_prices = initial_price * np.exp(cumulative_log_returns)
    cumulative_prices = pd.Series(cumulative_prices, index=df_log_ret.index, name='Cumulative_Price')

    return df_log_ret, cumulative_prices

def compute_variability_series(time_series, window_size, wavelet='haar', level=4):
    """
    Compute the normalized variability indicator for the entire time series.

    Parameters:
    - time_series: numpy array or pd.Series, the input financial time series
    - window_size: int, size of the moving window (default 32)
    - wavelet: str, wavelet type (default 'haar')
    - level: int, DWT decomposition level (default 4)

    Returns:
    - variability_series: pd.Series, normalized variability for each valid time point
    """
    half_window = window_size // 2
    start_t = half_window
    end_t = time_series.size - half_window + 1
    variability_series = []

    for t in range(start_t, end_t):
        try:
            variability_series.append(wavelet_decomposition_from_symmetrized_signal(
                time_series, t, window_size, wavelet, level
            ))
        except ValueError:
            continue  # Skip invalid indices

    # Convert to pandas Series with same index as input if available
    # Flat the variability_series to a 1D array
    variability_array = np.concatenate([np.ravel(v) for v in variability_series])
    variability_df = pd.DataFrame({'Variability': variability_array}, index=time_series.index[start_t:end_t-1])
    return variability_df

def wavelet_decomposition_from_symmetrized_signal(time_series, t, window_size, wavelet='haar', level=4):
    """
    Performs wavelet decomposition on a symmetrized signal around a given time index 't'.

    Parameters:
    - time_series: numpy array or pd.Series, the input financial time series (e.g., log returns).
    - t: int, the time index around which to symmetrize the signal.
    - window_size: int, size of the moving window (must be a power of 2, default 32).
    - wavelet (str): Type of wavelet to use (default: 'haar').
    - level (int): Maximum decomposition level.

    Returns:
    - coeffs (list): List of wavelet coefficients [cA_level, cD_level, ..., cD_1].
    """
    # Convert time_series to numpy array if it's a pandas Series
    time_series = np.asarray(time_series)

    # Step 1: Validate window_size
    if not (window_size & (window_size - 1) == 0):
        raise ValueError("Window size must be a power of 2.")
    half_window = window_size // 2

    # Step 2: Validate time index t
    if t < half_window or t >= len(time_series) - half_window + 1:
        raise ValueError("Time index t is out of bounds for the given time series and window size.")

    # Step 3: Symmetrize the signal around time t
    # Form: [x(t-15), ..., x(t-1), x(t), x(t-1), ..., x(t-15)] for window_size=32
    left = time_series[t - half_window:t]
    right = time_series[t + 1:t + half_window + 1][::-1]
    symmetrized_signal = np.concatenate([left, right])

    # Verify symmetrized signal length
    if len(symmetrized_signal) != window_size:
        raise ValueError(f"Symmetrized signal length ({len(symmetrized_signal)}) does not match window size ({window_size}).")

    # Step 4: Apply DWT to obtain coefficients
    # Using 'symmetric' mode to match the behavior of the original function's wavelet decomposition.
    coeffs = pywt.wavedec(symmetrized_signal, wavelet=wavelet, level=level, mode='symmetric')
    # coeffs[0] is A^level (approximation), coeffs[1] is D^level, coeffs[2] is D^(level-1), ...

    return coeffs[-level]

def plot_wavelet_coefficients(coeffs, level, ticker_name):
    """
    Genera y guarda gráficos de los coeficientes wavelet.

    Parameters:
    - coeffs (pd.DataFrame o list): Coeficientes wavelet o DataFrame con una columna.
    - level (int): Nivel máximo de descomposición.
    - ticker_name (str): Nombre del ticker para el nombre del archivo.
    """
    # Si coeffs es un DataFrame con una sola columna
    if isinstance(coeffs, pd.DataFrame) and coeffs.shape[1] == 1:
        plt.figure(figsize=(12, 3))
        plt.plot(coeffs.iloc[:, 0], label=coeffs.columns[0])
        plt.title(f'Wavelet Coefficient (Level {level})', fontsize=18)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Coefficient Value', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'plots/wavelet_coefficients_log_returns_{ticker_name}_{level}.png')
        plt.close()
    else:
        fig, axs = plt.subplots(len(coeffs), 1, figsize=(12, 10), sharex=False)
        axs[0].plot(coeffs[0])
        axs[0].set_title(f'Approximation Coefficients (cA{level})', fontsize=18)
        axs[0].tick_params(axis='both', labelsize=14)
        for i in range(1, len(coeffs)):
            axs[i].plot(coeffs[i])
            axs[i].set_title(f'Detail Coefficients (cD{level - i + 1})', fontsize=18)
            axs[i].tick_params(axis='both', labelsize=14)
        fig.suptitle('Wavelet Coefficients of Log Returns', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/wavelet_coefficients_log_returns_{ticker_name}.png')
        plt.close()

def extract_extreme_dates(dfs_coeffs, top_pct=5, min_initial_data=100):
    """
    Identifica fechas con oscilaciones extremas en los coeficientes wavelet, calculando el percentil dinámicamente
    usando solo datos pasados.

    Parameters:
    - dfs_coeffs (list): Lista de DataFrames, cada uno con una columna 'Variability' y un índice temporal.
    - top_pct (float): Porcentaje superior de oscilaciones a considerar.
    - min_initial_data (int): Número mínimo de datos iniciales para calcular el percentil (default: 100).

    Returns:
    - dict: Diccionario {coef_name: DatetimeIndex} con fechas extremas para cada nivel.
    """
    extreme_dates = {}
    
    for level, df in enumerate(dfs_coeffs, start=1):  # levels 1 to 4
        # Nombre del coeficiente (cD_level, ya que usas coeffs[-level])
        coef_name = f'cD{level}'
        
        # Obtener los valores de los coeficientes y el índice
        coeffs = df['Variability'].values
        base_index = df.index
        
        # Lista para almacenar las fechas extremas
        extreme_indices = []
        
        # Iterar sobre las fechas a partir de min_initial_data
        for i in range(min_initial_data, len(coeffs)):
            # Calcular el percentil usando solo los datos pasados (hasta i)
            past_data = np.abs(coeffs[:i+1])  # Incluye el dato actual
            thr = np.percentile(past_data, 100 - top_pct)
            
            # Verificar si el coeficiente actual es extremo
            if np.abs(coeffs[i]) >= thr:
                extreme_indices.append(i)
        
        # Convertir índices a fechas
        if extreme_indices:
            extreme_dates[coef_name] = base_index[extreme_indices]
        else:
            extreme_dates[coef_name] = pd.DatetimeIndex([])  # Índice vacío si no hay extremos
        
    return extreme_dates

def plot_extreme_dates(cumulative_prices, extreme_dates, ticker_name, top_pct):
    """
    Gráfica precios acumulados con líneas verticales en fechas extremas.

    Parameters:
    - cumulative_prices (pd.Series): Precios acumulados.
    - extreme_dates (dict): Diccionario de fechas extremas.
    - ticker_name (str): Nombre del ticker.
    - top_pct (float): Porcentaje superior usado.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cumulative_prices, linewidth=1.2)
    ax.set_title(f'Cumulative Price with Top {top_pct}% Wavelet Oscillations', fontsize=22)
    ax.set_ylabel('Price', fontsize=18)
    ax.set_xlabel('Date', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    
    palette = sns.color_palette('husl', n_colors=len(extreme_dates))
    for (coef_name, dates), color in zip(extreme_dates.items(), palette):
        for date in dates:
            ax.axvline(date, color=color, linewidth=1.0, linestyle='--', alpha=0.7)
    
    handles = [Line2D([0], [0], color=c, lw=3, linestyle='--', label=n) 
               for (n, _), c in zip(extreme_dates.items(), palette)]
    ax.legend(handles=handles, title='Coefficient band', loc='upper left', fontsize=14, title_fontsize=15)
    plt.tight_layout()
    plt.savefig(f'plots/dates_with_highest_coeff_{ticker_name}_{top_pct}.png')
    plt.close()

def plot_extreme_dates_with_coefficients(cumulative_prices, dfs_coeffs, extreme_dates, ticker_name, top_pct, min_initial_data=100):
    """
    Gráfica precios acumulados con líneas verticales en fechas extremas y subplots de coeficientes wavelet con umbrales dinámicos.

    Parameters:
    - cumulative_prices (pd.Series): Precios acumulados.
    - dfs_coeffs (list): Lista de DataFrames, cada uno con una columna 'Variability' y un índice temporal.
    - extreme_dates (dict): Diccionario de fechas extremas {coef_name: DatetimeIndex}.
    - ticker_name (str): Nombre del ticker.
    - top_pct (float): Porcentaje superior para umbrales.
    - min_initial_data (int): Número mínimo de datos iniciales para calcular el percentil (default: 100).
    """
    # Número de subplots: 1 para precios + número de niveles de detalles
    n_details = 4  # Número de niveles (cD1 a cD4)
    heights = [4] + [2] * n_details  # Altura 4 para precios, 2 para cada subplot de coeficientes
    fig = plt.figure(figsize=(14, 4 + 2 * n_details))
    gs = fig.add_gridspec(n_details + 1, 1, height_ratios=heights)
    axes = [fig.add_subplot(gs[i]) for i in range(n_details + 1)]
    
    # Gráfica precios acumulados
    ax_prices = axes[0]
    ax_prices.plot(cumulative_prices, linewidth=1.2)
    ax_prices.set_title(f'Cumulative Price with Top {top_pct}% Wavelet Oscillations for {ticker_name}', fontsize=22)
    ax_prices.set_ylabel('Price', fontsize=14)
    ax_prices.tick_params(axis='both', labelsize=12)
    
    # Añadir líneas verticales para fechas extremas
    palette = sns.color_palette('husl', n_colors=len(extreme_dates))
    for (coef_name, dates), color in zip(extreme_dates.items(), palette):
        for date in dates:
            ax_prices.axvline(date, color=color, linewidth=1.0, linestyle='--', alpha=0.7)
    
    # Leyenda para precios
    handles = [Line2D([0], [0], color=c, lw=3, linestyle='--', label=n) 
               for (n, _), c in zip(extreme_dates.items(), palette)]
    ax_prices.legend(handles=handles, title='Coefficient band', loc='upper left', fontsize=12, title_fontsize=13)
    
    # Graficar coeficientes wavelet con umbrales dinámicos
    for i, df in enumerate(dfs_coeffs, start=1):  # Niveles 1 a 4
        ax = axes[i]
        coef_name = f'cD{i}'
        coeffs = df['Variability'].values
        coef_dates = df.index
        
        # Graficar coeficientes
        color = palette[i - 1]  # Usar el mismo color que las líneas verticales
        ax.plot(coef_dates, coeffs, label=coef_name, color=color)
        
        # Calcular y graficar umbrales dinámicos
        thresholds = []
        for j in range(min_initial_data, len(coeffs)):
            past_data = np.abs(coeffs[:j+1])
            thr = np.percentile(past_data, 100 - top_pct)
            thresholds.append(thr)
        # Extender umbrales con NaN para los primeros min_initial_data puntos
        thresholds = [np.nan] * min_initial_data + thresholds
        ax.plot(coef_dates, thresholds, color='red', linestyle='--', label='Threshold')
        ax.plot(coef_dates, [-t if not np.isnan(t) else np.nan for t in thresholds], color='red', linestyle='--')
        
        ax.set_ylabel(f'{coef_name}', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc='upper right', fontsize=10)
        if i != n_details:
            ax.tick_params(axis='x', which='both', labelbottom=False)
    
    # Ajustar etiquetas y título
    axes[-1].set_xlabel('Date', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'plots/dates_with_highest_coeff_and_subplots_{ticker_name}_{top_pct}.png')
    plt.close()

def compute_H_series(log_returns, coeffs, level):
    """
    Calcula las series H_i(t) para diferentes niveles.

    Parameters:
    - log_returns (pd.Series): Serie de log returns.
    - coeffs (list): Coeficientes wavelet.
    - level (int): Nivel máximo de descomposición.

    Returns:
    - H_series (dict): Diccionario con series H_i(t).
    - detail_reconstructed_series (dict): Series de detalles reconstruidos.
    """
    H_series = {}
    detail_reconstructed_series = {}
    for i in range(1, level + 1):
        abs_detail_sum = np.zeros_like(log_returns.values, dtype=float)
        detail_reconstructed_series[i] = {}
        for j in range(1, i + 1):
            idx_coeff = level - j + 1
            rec_coeff_list = [np.zeros_like(coeffs[0])] + [
                coeffs[k] if k == idx_coeff else np.zeros_like(coeffs[k]) for k in range(1, len(coeffs))]
            detail_rec = pywt.waverec(rec_coeff_list, 'haar', mode='periodization')[:len(abs_detail_sum)]
            abs_detail_sum += np.abs(detail_rec)
            detail_reconstructed_series[i][j] = detail_rec
        
        approx_coeff_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approx = pywt.waverec(approx_coeff_list, 'haar', mode='periodization')[:len(abs_detail_sum)]
        approx = np.where(approx == 0, 1e-10, approx)
        H_series[i] = abs_detail_sum / approx
    return H_series, detail_reconstructed_series

def plot_H_series_with_details(H_series, detail_reconstructed_series, cumulative_prices, ticker_name, level):
    """
    Genera gráficos de H_i(t) con detalles y precios acumulados.

    Parameters:
    - H_series (dict): Series H_i(t).
    - detail_reconstructed_series (dict): Series de detalles reconstruidos.
    - cumulative_prices (pd.Series): Precios acumulados.
    - ticker_name (str): Nombre del ticker.
    - level (int): Nivel máximo de descomposición.
    """
    for i in range(1, level + 1):
        n_subplots = i + 1
        fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 4 * n_subplots), sharex=True)
        if n_subplots == 1:
            axes = [axes]
        
        for j in range(1, i + 1):
            ax1 = axes[j - 1]
            ax2 = ax1.twinx()
            detail_series = pd.Series(detail_reconstructed_series[i][j], index=cumulative_prices.index)
            sns.lineplot(data=detail_series, ax=ax1, color="green", label=f"Detail D{j}", alpha=0.65)
            ax2.plot(cumulative_prices.index, cumulative_prices, label="Cumulative Price")
            ax1.set_ylabel(f"D{j} Amplitude", fontsize=14)
            ax2.set_ylabel("Cumulative Price", fontsize=14)
            ax1.tick_params(axis='y', labelsize=12)
            ax2.tick_params(axis='y', labelsize=12)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)
        
        ax1 = axes[-1]
        ax2 = ax1.twinx()
        h_series = pd.Series(H_series[i], index=cumulative_prices.index)
        sns.lineplot(data=h_series, ax=ax1, color="red", label=f"H_{i}(t)", alpha=0.65)
        ax2.plot(cumulative_prices.index, cumulative_prices, label="Cumulative Price")
        ax1.set_ylabel(f"H_{i} Value (%)", fontsize=14)
        ax2.set_ylabel("Cumulative Price", fontsize=14)
        ax1.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)
        
        axes[-1].set_xlabel("Date", fontsize=14)
        axes[-1].tick_params(axis='x', labelsize=12)
        fig.suptitle(f"Reconstructed Details and H_{i} with Cumulative Price for {ticker_name}", fontsize=18)
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/H_{i}_with_details_and_price_{ticker_name}.png')
        plt.close()

def compute_mobile_energy(H_series, window_size=16):
    """
    Calcula la energía móvil para cada H_i(t).

    Parameters:
    - H_series (dict): Series H_i(t).
    - window_size (int): Tamaño de la ventana móvil.

    Returns:
    - energy_series (dict): Series de energía móvil.
    """
    energy_series = {}
    for i in H_series:
        h_series = pd.Series(H_series[i])
        energy_series[i] = h_series.rolling(window=window_size, min_periods=1).apply(lambda x: np.sum(x**2), raw=True).values
    return energy_series

def plot_mobile_energy(energy_series, cumulative_prices, ticker_name, window_size):
    """
    Grafica la energía móvil junto con los precios acumulados.

    Parameters:
    - energy_series (dict): Series de energía móvil.
    - cumulative_prices (pd.Series): Precios acumulados.
    - ticker_name (str): Nombre del ticker.
    - window_size (int): Tamaño de la ventana móvil.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    for i in energy_series:
        ax1.plot(cumulative_prices.index, energy_series[i], label=f'$Energy(H_{i})$')
    ax1.set_xlabel('Fecha', fontsize=18)
    ax1.set_ylabel('Energía Móvil de $H_i(t)$', fontsize=18)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_title(f'Energía Móvil de $H_i(t)$ para cada Nivel (Ventana {window_size} días)', fontsize=22)
    
    ax2 = ax1.twinx()
    ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return')
    ax2.set_ylabel('Cumulative Return', fontsize=18)
    ax2.tick_params(axis='y', labelsize=15)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/Energy_series_all_levels_{ticker_name}.png')
    plt.close()

def denoise_H_series(H_series, level, threshold_energy_retain=0.4):
    """
    Aplica denoising a las series H_i(t).

    Parameters:
    - H_series (dict): Series H_i(t).
    - level (int): Nivel máximo de descomposición.
    - threshold_energy_retain (float): Fracción de energía a retener.

    Returns:
    - denoised_series (dict): Series H_i(t) denoised.
    """
    denoised_series = {}
    for i in H_series:
        coeffs_H = pywt.wavedec(H_series[i], 'haar', level=level, mode='periodization')
        flat_coeffs = np.concatenate([c for c in coeffs_H])
        sorted_abs_coeffs = np.sort(np.abs(flat_coeffs))[::-1]
        total_energy = np.sum(sorted_abs_coeffs**2)
        if total_energy == 0:
            T = 0
        else:
            cumulative_energy = np.cumsum(sorted_abs_coeffs**2)
            idx_threshold = np.searchsorted(cumulative_energy, threshold_energy_retain * total_energy)
            T = sorted_abs_coeffs[idx_threshold] if idx_threshold < len(sorted_abs_coeffs) else sorted_abs_coeffs[-1]
        thresholded_coeffs = [pywt.threshold(c, T, mode='hard') for c in coeffs_H]
        denoised_series[i] = pywt.waverec(thresholded_coeffs, 'haar', mode='periodization')[:len(H_series[i])]
    return denoised_series

def plot_denoised_H_series(denoised_series, cumulative_prices, ticker_name, threshold_energy_retain):
    """
    Grafica todas las series H_i(t) denoised.

    Parameters:
    - denoised_series (dict): Series H_i(t) denoised.
    - cumulative_prices (pd.Series): Precios acumulados.
    - ticker_name (str): Nombre del ticker.
    - threshold_energy_retain (float): Fracción de energía retenida.
    """
    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.gca()
    for i in denoised_series:
        ax1.plot(cumulative_prices.index[:len(denoised_series[i])], denoised_series[i], label=f'Denoised $H_{i}$')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Amplitud Denoised $H_i(t)$')
    ax1.set_title(f'Denoised $H_i(t)$ para cada Nivel ({threshold_energy_retain*100}% Energía Retenida)')
    ax1.tick_params(axis='y', labelsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(cumulative_prices.index, cumulative_prices.values, label='Cumulative Return', color='black', alpha=0.5)
    ax2.set_ylabel('Cumulative Return')
    ax2.tick_params(axis='y', labelsize=12)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/denoised_H_series_all_levels_{ticker_name}.png')
    plt.close()

def identify_critical_dates(denoised_series, log_returns):
    """
    Identifica fechas críticas basadas en las series denoised.

    Parameters:
    - denoised_series (dict): Series H_i(t) denoised.
    - log_returns (pd.Series): Serie de log returns.

    Returns:
    - critical_dates (dict): Diccionario con fechas críticas por nivel.
    """
    critical_dates = {}
    for i in denoised_series:
        critical_indices = np.where(np.abs(denoised_series[i]) > 1e-10)[0]
        critical_dates[i] = log_returns.index[critical_indices]
    return critical_dates

def early_warning_system(log_returns, critical_dates, level, threshold_energy_retain=0.4, counter_threshold=10):
    """
    Implementa un sistema de contador para señales de alerta robustas.

    Parameters:
    - log_returns (pd.Series): Serie de log returns.
    - critical_dates (dict): Fechas críticas por nivel.
    - level (int): Nivel máximo de descomposición.
    - threshold_energy_retain (float): Fracción de energía retenida.
    - counter_threshold (int): Umbral de contador para considerar una alerta.

    Returns:
    - early_warnings (dict): Fechas de alerta temprana por nivel.
    """
    early_warnings = {}
    for i in range(1, level + 1):
        early_warnings[i] = []
        if not critical_dates[i].empty:
            for crit_date in critical_dates[i]:
                idx = log_returns.index.get_loc(crit_date)
                counter = 0
                for x in range(max(0, idx - 15), min(len(log_returns), idx + 16)):
                    truncated = log_returns.iloc[:x+1].values
                    if not truncated.size:
                        continue
                    flipped = truncated[::-1]
                    pattern_len = 2 * len(truncated)
                    if pattern_len == 0:
                        continue
                    num_repeats = (len(log_returns) // pattern_len) + 1
                    modified = np.concatenate([truncated, flipped] * num_repeats)[:len(log_returns)]
                    
                    coeffs_mod = pywt.wavedec(modified, 'haar', level=level, mode='periodization')
                    details_sum_mod = np.zeros_like(log_returns.values, dtype=float)
                    for j in range(1, i + 1):
                        idx_Dj = level - j + 1
                        coeff_list = [np.zeros_like(coeffs_mod[0])] + [
                            coeffs_mod[k] if k == idx_Dj else np.zeros_like(coeffs_mod[k]) for k in range(1, len(coeffs_mod))]
                        detail_rec = pywt.waverec(coeff_list, 'haar', mode='periodization')[:len(details_sum_mod)]
                        details_sum_mod += np.abs(detail_rec)
                    
                    approx_list = [coeffs_mod[0]] + [np.zeros_like(c) for c in coeffs_mod[1:]]
                    approx_mod = pywt.waverec(approx_list, 'haar', mode='periodization')[:len(details_sum_mod)]
                    approx_mod = np.where(approx_mod == 0, 1e-10, approx_mod)
                    H_mod = details_sum_mod / approx_mod
                    
                    coeffs_H_mod = pywt.wavedec(H_mod, 'haar', level=level, mode='periodization')
                    flat_coeffs_mod = np.concatenate([c for c in coeffs_H_mod])
                    sorted_abs_coeffs_mod = np.sort(np.abs(flat_coeffs_mod))[::-1]
                    total_energy = np.sum(sorted_abs_coeffs_mod**2)
                    if total_energy == 0:
                        T_mod = 0
                    else:
                        cumulative_energy = np.cumsum(sorted_abs_coeffs_mod**2)
                        idx_threshold = np.searchsorted(cumulative_energy, threshold_energy_retain * total_energy)
                        T_mod = sorted_abs_coeffs_mod[idx_threshold] if idx_threshold < len(sorted_abs_coeffs_mod) else sorted_abs_coeffs_mod[-1]
                    
                    thresholded_coeffs = [pywt.threshold(c, T_mod, mode='hard') for c in coeffs_H_mod]
                    denoised_mod = pywt.waverec(thresholded_coeffs, 'haar', mode='periodization')[:len(H_mod)]
                    if idx < len(denoised_mod) and np.abs(denoised_mod[idx]) > 1e-10:
                        counter += 1
                
                if counter > counter_threshold:
                    early_warnings[i].append(crit_date)
    return early_warnings

def plot_early_warnings(cumulative_prices, early_warnings, ticker_name, level_to_plot):
    """
    Gráfica precios acumulados con señales de alerta.

    Parameters:
    - cumulative_prices (pd.Series): Precios acumulados.
    - early_warnings (dict): Fechas de alerta temprana.
    - ticker_name (str): Nombre del ticker.
    - level_to_plot (int): Nivel de H_i a graficar.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(cumulative_prices.index, cumulative_prices, label=f'Precios Acumulados ({ticker_name})', color='blue', alpha=0.7)
    if level_to_plot in early_warnings and early_warnings[level_to_plot]:
        for idx, date in enumerate(early_warnings[level_to_plot]):
            plt.axvline(x=date, color='red', linestyle='--', linewidth=1.2, 
                        label=f'Alerta Temprana H_{level_to_plot}' if idx == 0 else None)
    plt.title(f'Precios Acumulados y Señales de Alerta Temprana para $H_{level_to_plot}$')
    plt.xlabel('Fecha')
    plt.ylabel('Precio Acumulado')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(f'plots/cumulative_prices_with_H{level_to_plot}_warnings_{ticker_name}.png')
    plt.close()

def plot_critical_dates_frequency(critical_dates, early_warnings, ticker_name, level):
    """
    Grafica la frecuencia de fechas críticas detectadas.

    Parameters:
    - critical_dates (dict): Fechas críticas por nivel.
    - early_warnings (dict): Fechas de alerta temprana.
    - ticker_name (str): Nombre del ticker.
    - level (int): Nivel máximo de descomposición.
    """
    plt.figure(figsize=(16, 8))
    colors = ['blue', 'green', 'purple', 'orange']
    for i in range(1, level + 1):
        if not critical_dates[i].empty:
            dates = critical_dates[i]
            counts = [sum(1 for _ in range(1, level + 1) if date in critical_dates[i]) for date in dates]
            for date, count in zip(dates, counts):
                is_alert = date in early_warnings.get(i, [])
                color = 'red' if is_alert else colors[i-1]
                marker = 'o' if is_alert else 'x'
                size = 50 if is_alert else 30
                label = f'Alerta Temprana (H_{i}, Contador > 10)' if is_alert else f'Fecha Crítica (H_{i}, Contador <= 10)'
                plt.scatter(date, count, color=color, marker=marker, s=size, label=label, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    plt.axhline(y=10, color='black', linestyle='--', linewidth=1.2, label='Umbral de Alerta (Contador = 10)')
    plt.title('Validación de Fechas Críticas: Robustez y Señales de Alerta')
    plt.xlabel('Fecha')
    plt.ylabel('Contador de Robustez')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'plots/frequency_critical_dates_robustness_check_{ticker_name}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    ticker_name = 'SP'
    file_path = 'sector_log_returns.csv'
    level = 4
    top_pct = 1
    window_size = 2**level
    threshold_energy_retain = 0.4
    
    # Plot coefficients for four levels
    log_returns, cumulative_prices = load_and_prepare_data(file_path, 'S&P 500')
    dfs_coeffs = []
    for level in range(1, 5):
        window_size = 2**level
        df_coeffs = compute_variability_series(log_returns, window_size=window_size, wavelet='haar', level=level)
        plot_wavelet_coefficients(df_coeffs, level, ticker_name)
        dfs_coeffs.append(df_coeffs)

    # Combine coefficients into a single DataFrame
    df_coeffs = pd.concat(dfs_coeffs, axis=1)
    df_coeffs.columns = [f'cD{level - i + 1}' for i in range(1, level + 1)]
    # Obtener fechas extremas
    extreme_dates = extract_extreme_dates(dfs_coeffs, top_pct=3, min_initial_data=100)

    # Imprimir resultados
    for coef_name, dates in extreme_dates.items():
        print(f"{coef_name}: {dates}")
    plot_extreme_dates(cumulative_prices, extreme_dates, ticker_name, top_pct)
    # Graficar precios y coeficientes
    plot_extreme_dates_with_coefficients(cumulative_prices, dfs_coeffs, extreme_dates, ticker_name, top_pct, min_initial_data=100)
    
    H_series, detail_series = compute_H_series(log_returns, df_coeffs, level)
    plot_H_series_with_details(H_series, detail_series, cumulative_prices, ticker_name, level)
    energy_series = compute_mobile_energy(H_series, window_size)
    plot_mobile_energy(energy_series, cumulative_prices, ticker_name, window_size)
    denoised_series = denoise_H_series(H_series, level, threshold_energy_retain)
    plot_denoised_H_series(denoised_series, cumulative_prices, ticker_name, threshold_energy_retain)
    critical_dates = identify_critical_dates(denoised_series, log_returns)
    early_warnings = early_warning_system(log_returns, critical_dates, level, threshold_energy_retain)
    plot_early_warnings(cumulative_prices, early_warnings, ticker_name, level)
    plot_critical_dates_frequency(critical_dates, early_warnings, ticker_name, level)