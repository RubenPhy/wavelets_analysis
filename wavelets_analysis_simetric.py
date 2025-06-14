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
    
    n = len(df_log_ret)
    n_adjusted = 2**int(np.log2(n))
    return df_log_ret.iloc[:n_adjusted], cumulative_prices.iloc[:n_adjusted]

def compute_normalized_variability(time_series, t, window_size=32, wavelet='haar', level=4):
    """
    Compute the normalized variability indicator for a financial time series at time t
    using Haar DWT at specified level with a given window size, as per the theoretical framework.

    Parameters:
    - time_series: numpy array or pd.Series, the input financial time series (e.g., log returns)
    - t: int, the time index at which to compute the indicator
    - window_size: int, size of the moving window (must be a power of 2, default 32)
    - wavelet: str, wavelet type (default 'haar')
    - level: int, DWT decomposition level (default 4)

    Returns:
    - normalized_variability: float, the normalized variability indicator at time t
    """
    # Convert time_series to numpy array if it's a pandas Series
    time_series = np.asarray(time_series)
    
    # Step 1: Select a moving window (power of 2)
    if not (window_size & (window_size - 1) == 0):
        raise ValueError("Window size must be a power of 2.")
    half_window = window_size // 2

    # Validate time index t
    if t < half_window or t >= len(time_series) - half_window + 1:
        raise ValueError("Time index t is out of bounds for the given time series and window size.")

    # Step 2: Symmetrize the signal around time t
    # Form: [x(t-15), ..., x(t-1), x(t), x(t), x(t-1), ..., x(t-15)] for window_size=32
    left = time_series[t - half_window:t]  # x(t-15), ..., x(t-1)
    right = time_series[t - half_window + 1:t + 1][::-1]  # x(t), x(t-1), ..., x(t-15)
    symmetrized_signal = np.concatenate([left, right])

    # Verify symmetrized signal length
    if len(symmetrized_signal) != window_size:
        raise ValueError(f"Symmetrized signal length ({len(symmetrized_signal)}) does not match window size ({window_size}).")

    # Step 3: Apply Haar DWT to obtain detail coefficients at specified level
    coeffs = pywt.wavedec(symmetrized_signal, wavelet=wavelet, level=level, mode='symmetric')
    # coeffs[0] is A^level (approximation), coeffs[1] is D^level, coeffs[2] is D^(level-1), ...
    D_level = coeffs[1]  # Detail coefficients at specified level

    # Compute variability of D^level using: (1/(k2 - k1 + 1)) * sum(|x_{j+1} - x_j|)
    if len(D_level) > 1:
        variability = np.sum(np.abs(np.diff(D_level))) / len(D_level)
    else:
        variability = 0  # Handle case where D_level is too short

    # Step 4: Normalize variability with the mean of A^level (approximation coefficients)
    A_level = coeffs[0]  # Approximation coefficients at specified level
    A_level_mean = np.mean(np.abs(A_level)) if len(A_level) > 0 else 1  # Use absolute mean

    # Compute normalized variability
    normalized_variability = variability / A_level_mean if A_level_mean != 0 else 0

    return normalized_variability

def compute_variability_series(time_series, window_size=32, wavelet='haar', level=4):
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
    variability_series = np.zeros(time_series.size)  # Initialize with zeros
    variability_series[:] = np.nan  # Initialize with NaN for invalid indices

    for t in range(start_t, end_t):
        try:
            variability_series[t] = compute_normalized_variability(
                time_series, t, window_size, wavelet, level
            )
        except ValueError:
            continue  # Skip invalid indices

    # Convert to pandas Series with same index as input if available
    return pd.Series(variability_series, index=time_series.index, name='Variability')

def plot_variability_and_prices(log_returns, cumulative_prices, variability_series, ticker_name='SP'):
    """
    Plot the log returns, cumulative prices, and normalized variability series.

    Parameters:
    - log_returns: pd.Series, log returns of the financial time series
    - cumulative_prices: pd.Series, cumulative prices
    - variability_series: pd.Series, normalized variability series
    - ticker_name: str, name of the ticker (default 'SP')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot cumulative prices
    ax1.plot(cumulative_prices.index, cumulative_prices, label='Cumulative Price', color='blue')
    ax1.set_title(f'Cumulative Prices ({ticker_name})')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot variability series
    ax2.plot(variability_series.index, variability_series, label='Normalized Variability', color='red')
    ax2.set_title(f'Normalized Variability (DWT Level {level}, Window Size {window_size})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Variability')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def compute_all_variability_series(time_series, window_size=32, wavelet='haar', level=4):
    """
    Compute the normalized variability indicator for the entire time series at all levels up to the specified level.
    """
    time_series = np.asarray(time_series)
    half_window = window_size // 2
    start_t = half_window
    end_t = len(time_series) - half_window + 1
    variability_series = {}

    # Initialize series with proper index if time_series is a pd.Series
    if isinstance(time_series, pd.Series):
        index = time_series.index
        variability_series = {f'D{i}': pd.Series(np.full(len(time_series), np.nan), index=index) for i in range(1, level + 1)}
    else:
        variability_series = {f'D{i}': np.full(len(time_series), np.nan) for i in range(1, level + 1)}

    for t in range(start_t, end_t):
        try:
            # Compute DWT coefficients for the symmetrized window
            left = time_series[t - half_window:t]
            right = time_series[t - half_window + 1:t + 1][::-1]
            symmetrized_signal = np.concatenate([left, right])
            coeffs = pywt.wavedec(symmetrized_signal, wavelet=wavelet, level=level, mode='symmetric')
            for i in range(1, level + 1):
                D_i = coeffs[i]  # Detail coefficients at level i
                if len(D_i) > 1:
                    variability = np.sum(np.abs(np.diff(D_i))) / len(D_i)
                else:
                    variability = 0
                A_i = coeffs[0] if i == 1 else coeffs[i - 1]  # Approximation for normalization
                A_i_mean = np.mean(np.abs(A_i)) if len(A_i) > 0 else 1
                normalized_variability = variability / A_i_mean if A_i_mean != 0 else 0
                if isinstance(time_series, pd.Series):
                    variability_series[f'D{i}'][t] = normalized_variability
                else:
                    variability_series[f'D{i}'][t] = normalized_variability
        except ValueError:
            continue

    if isinstance(time_series, pd.Series):
        for key in variability_series:
            variability_series[key] = pd.Series(variability_series[key], index=time_series.index, name=key)
    else:
        for key in variability_series:
            variability_series[key] = pd.Series(variability_series[key], name=key)

    return variability_series


def plot_dwt_levels_with_threshold(log_returns, cumulative_prices, variability_series, ticker_name='SP', percentile_threshold=95):
    """
    Plot the cumulative prices and DWT detail coefficients (D1, D2, D3, D4) with a percentile-based threshold
    and vertical lines for threshold crossings.
    """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # Plot cumulative prices (coefficient band) in black
    ax1.plot(cumulative_prices.index, cumulative_prices, label='Coefficient Band', color='black')
    ax1.set_title(f'Cumulative Prices ({ticker_name})')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')

    # Plot D1, D2, D3, D4 variability series with individual thresholds
    ax2.plot(cumulative_prices.index, variability_series['D1'], label='D1', color='red')
    threshold_d1 = np.percentile(variability_series['D1'].dropna(), percentile_threshold)
    ax2.axhline(y=threshold_d1, color='black', linestyle='--', label=f'Threshold D1 ({percentile_threshold}th percentile)')
    ax2.set_title('D1 Variability')
    ax2.set_ylabel('Variability')
    ax2.legend(loc='best')

    ax3.plot(cumulative_prices.index, variability_series['D2'], label='D2', color='green')
    threshold_d2 = np.percentile(variability_series['D2'].dropna(), percentile_threshold)
    ax3.axhline(y=threshold_d2, color='black', linestyle='--', label=f'Threshold D2 ({percentile_threshold}th percentile)')
    ax3.set_title('D2 Variability')
    ax3.set_ylabel('Variability')
    ax3.legend(loc='best')

    ax4.plot(cumulative_prices.index, variability_series['D3'], label='D3', color='cyan')
    threshold_d3 = np.percentile(variability_series['D3'].dropna(), percentile_threshold)
    ax4.axhline(y=threshold_d3, color='black', linestyle='--', label=f'Threshold D3 ({percentile_threshold}th percentile)')
    ax4.set_title('D3 Variability')
    ax4.set_ylabel('Variability')
    ax4.legend(loc='best')

    ax5.plot(cumulative_prices.index, variability_series['D4'], label='D4', color='purple')
    threshold_d4 = np.percentile(variability_series['D4'].dropna(), percentile_threshold)
    ax5.axhline(y=threshold_d4, color='black', linestyle='--', label=f'Threshold D4 ({percentile_threshold}th percentile)')
    ax5.set_title('D4 Variability')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Variability')
    ax5.legend(loc='best')

    # Find indices where variability exceeds threshold for each level with distinct lines
    thresholds = {'D1': threshold_d1, 'D2': threshold_d2, 'D3': threshold_d3, 'D4': threshold_d4}
    colors = {'D1': 'red', 'D2': 'green', 'D3': 'cyan', 'D4': 'purple'}

    for level, threshold in thresholds.items():
        crossings = variability_series[level][variability_series[level] > threshold].index
        for date in sorted(crossings):
            ax1.axvline(x=cumulative_prices.index[date], color=colors[level], linestyle='--', alpha=0.5, label=f'{level} Crossing' if date == min(crossings) else "")

    # Set x-axis limits to match cumulative_prices
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(cumulative_prices.index[0], cumulative_prices.index[-1])

    # Adjust legend to avoid duplication
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker_name = 'SP'
    file_path = 'sector_log_returns.csv'
    level = 4
    top_pct = 3
    window_size = 32
    threshold_energy_retain = 0.4
    percentile_threshold = 98
    
    log_returns, cumulative_prices = load_and_prepare_data(file_path, 'S&P 500')
    
    # Compute variability series (level 4, window size 32)
    variability_series = compute_variability_series(log_returns, window_size=window_size, level=level)

    # Plot results
    plot_variability_and_prices(log_returns, cumulative_prices, variability_series, ticker_name)

    # Alternative: Compute with level 3, window size 16
    level_alt = 3
    window_size_alt = 16
    variability_series_alt = compute_variability_series(log_returns, window_size=window_size_alt, level=level_alt)

    # Plot alternative results
    plot_variability_and_prices(log_returns, cumulative_prices, variability_series_alt, ticker_name)

    # Compute variability series for all levels up to 4
    variability_series = compute_all_variability_series(log_returns, window_size=window_size, level=level)

    # Plot results
    plot_dwt_levels_with_threshold(log_returns, cumulative_prices, variability_series, ticker_name, percentile_threshold)