import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os

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
    #plt.show()
    plt.close()

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
    # Set height ratios: ax1 is double the height of the others
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]}
    )

    # Plot cumulative prices (coefficient band) in black
    ax1.plot(cumulative_prices.index, cumulative_prices, label='Coefficient Band', color='black')
    ax1.set_title(f'Returns and Wavelet Coef. for {ticker_name} - Threshold: {percentile_threshold}th', fontsize=20, fontweight='bold')

    ax1.set_ylabel('Price')
    ax1.legend(loc='best')

    # Use seaborn 'husl' color palette for D1-D4
    husl_colors = sns.color_palette('husl', 4)
    dwt_levels = ['D1', 'D2', 'D3', 'D4']
    axes = [ax2, ax3, ax4, ax5]

    thresholds = {}
    for i, (level, ax) in enumerate(zip(dwt_levels, axes)):
        color = husl_colors[i]
        ax.plot(cumulative_prices.index, variability_series[level], label=level, color=color)
        threshold = np.percentile(variability_series[level].dropna(), percentile_threshold)
        thresholds[level] = threshold
        ax.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold {level} ({percentile_threshold}th percentile)')
        ax.set_title(f'{level} Variability', fontsize=16)
        ax.set_ylabel('Variability')
    ax5.set_xlabel('Date')

    # Find indices where variability exceeds threshold for each level with distinct lines
    for i, level in enumerate(dwt_levels):
        color = husl_colors[i]
        crossings = variability_series[level][variability_series[level] > thresholds[level]].index
        for date in sorted(crossings):
            ax1.axvline(x=cumulative_prices.index[date], color=color, linestyle='--', alpha=0.5, label=f'{level} Crossing' if date == min(crossings) else "")

    # Set x-axis limits to match cumulative_prices
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(cumulative_prices.index[0], cumulative_prices.index[-1])

    # Adjust legend to avoid duplication
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    # Save the plot as PNG in the 'plots' folder
    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', f'{ticker_name}_dwt_levels_threshold_{percentile_threshold}th.png'))
    #plt.show()
    plt.close()

def plot_d4_peak_returns_bins(log_returns, variability_series, percentile_threshold=95, window=15, sector='SP'):
    """
    Plot the mean returns in bins from d-15 to d+15 around D4 peak dates.

    Parameters:
    - log_returns: pd.Series, log returns of the financial time series
    - variability_series: pd.Series, normalized variability series for D4
    - percentile_threshold: int, percentile threshold for identifying peaks (default 95)
    - window: int, number of days before and after the peak to analyze (default 15)
    """
    # Compute threshold for D4
    threshold_d4 = np.percentile(variability_series.dropna(), percentile_threshold)
    
    # Identify peak days where D4 exceeds threshold
    peak_dates = variability_series[variability_series > threshold_d4].index
    
    # Initialize array to store returns for each relative day
    relative_returns = np.zeros(2 * window +1)
    count_days = np.zeros(2 * window + 1)
    
    for peak_idx in peak_dates:
        # Ensure window fits within data bounds
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(log_returns), peak_idx + window + 1)
        
        window_returns = log_returns.iloc[start_idx:end_idx]
        if len(window_returns) >= (2 * window + 1):
            for i, ret in enumerate(window_returns):
                rel_day = i - window  # Relative day from -window to +window
                if 0 <= rel_day + window < 2 * window + 1:
                    relative_returns[rel_day + window] += ret
                    count_days[rel_day + window] += 1
    
    # Compute mean returns for each bin
    mean_returns = relative_returns / count_days
    days = np.arange(-window, window + 1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(days, mean_returns, color='blue', width=0.8)
    plt.axvline(x=0, color='red', linestyle='--', label='Peak Day')
    plt.title(f'Mean Returns Around D4 Peaks (d-{window} to d+{window})\nSector: {sector}, Threshold: {percentile_threshold}th Percentile')
    plt.xlabel('Days Relative to Peak (d)')
    plt.ylabel('Mean Log Return')
    plt.legend()
    plt.grid(True)
    # Save the plot as PNG in the 'plots' folder, including sector name and percentile in the filename
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', f'd4_peak_returns_bins_{sector}_{percentile_threshold}th.png'))
    #plt.show()
    plt.close()

def plot_returns_with_events(cumulative_prices, variability_series, ticker_name='SP', percentile_threshold=95, events=None):
    """
    Plot cumulative prices with vertical lines for events and D4 peak dates.

    Parameters:
    - cumulative_prices: pd.Series, cumulative prices of the financial time series
    - variability_series: dict, variability series for all levels (e.g., from compute_all_variability_series)
    - ticker_name: str, name of the ticker (default 'SP')
    - percentile_threshold: int, percentile threshold for identifying peaks (default 95)
    - events: dict, dictionary of dates and corresponding event descriptions (default None)
    """
    if events is None:
        events = {
            "2020-03-16": "Panic due to COVID-19 pandemic",
            "2019-08-23": "Escalation of trade war with China",
            "2018-12-24": "Drop due to Fed fears",
            "2018-02-05": "Volatility collapse (\"Volmageddon\")",
            "2017-11-29": "Massive tech sector sell-off",
            "2016-06-24": "Surprise from Brexit result",
            "2015-08-24": "Concerns over Chinese economy",
            "2014-12-01": "Worry over oil price drop",
            "2013-06-20": "Announcement of stimulus withdrawal",
            "2012-12-31": "Agreement to avoid \"fiscal cliff\""
        }

    # Compute D4 threshold and peak dates
    d4_series = variability_series['D4']
    threshold_d4 = np.percentile(d4_series.dropna(), percentile_threshold)
    peak_dates = d4_series[d4_series > threshold_d4].index

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot cumulative prices
    ax1.plot(cumulative_prices.index, cumulative_prices, label='Coefficient Band', color='black')
    ax1.set_title(f'Returns and Wavelet Coef. for {ticker_name} - Threshold: {percentile_threshold}th', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')

    # Add vertical lines for events
    for date_str in events.keys():
        date = pd.to_datetime(date_str)
        ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.7, label='Event' if date == pd.to_datetime(list(events.keys())[0]) else "")

    # Add vertical lines for D4 peaks
    for date in peak_dates:
        ax1.axvline(x=cumulative_prices.index[date], color='red', linestyle='--', alpha=0.7, label='D4 Peak' if date == min(peak_dates) else "")

    # Set x-axis limits to match cumulative_prices
    ax1.set_xlim(cumulative_prices.index[0], cumulative_prices.index[-1])

    # Adjust legend to avoid duplication
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    # Save the plot as PNG in the 'plots' folder
    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', f'returns_with_events_{ticker_name}_{percentile_threshold}th.png'))
    #plt.show()
    plt.close()

if __name__ == "__main__":
    ticker_name = 'SP'
    file_path = 'sector_log_returns.csv'
    level = 4
    top_pct = 1
    window_size = 32
    percentile_threshold = 100 - top_pct
    
    # log_returns, cumulative_prices = load_and_prepare_data(file_path, 'S&P 500')
    
    # # Compute variability series (level 4, window size 32)
    # variability_series = compute_variability_series(log_returns, window_size=window_size, level=level)

    # # Plot results
    # plot_variability_and_prices(log_returns, cumulative_prices, variability_series, ticker_name)

    # # Alternative: Compute with level 3, window size 16
    # level_alt = 3
    # window_size_alt = 16
    # variability_series_alt = compute_variability_series(log_returns, window_size=window_size_alt, level=level_alt)

    # # Plot alternative results
    # plot_variability_and_prices(log_returns, cumulative_prices, variability_series_alt, ticker_name)

    # Load data for all sectors
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    sectors = [col for col in df.columns if col != 'log_return']
    
    for sector in sectors:
        print(f"Processing sector: {sector}")
        log_returns, cumulative_prices = load_and_prepare_data(file_path, column_name=sector)
        
        # Compute variability series for all levels up to 4
        variability_series = compute_all_variability_series(log_returns, window_size=window_size, level=level)

        # Plot and save results for each sector
        plot_dwt_levels_with_threshold(log_returns, cumulative_prices, variability_series, ticker_name=sector, percentile_threshold=percentile_threshold)

        # Plot D4 peak returns bins for each sector
        plot_d4_peak_returns_bins(log_returns, variability_series['D4'], percentile_threshold=percentile_threshold, sector=sector, window=int(window_size/2))

        # Plot critical dates specifically for S&P 500 sector
        if sector == sectors[0]:  # Assuming the first sector is S&P 500
            log_returns_sp, cumulative_prices_sp = load_and_prepare_data(file_path, column_name=sector)
            variability_series_sp = compute_all_variability_series(log_returns_sp, window_size=window_size, level=level)
            plot_returns_with_events(cumulative_prices_sp, variability_series_sp, ticker_name=sector, percentile_threshold=percentile_threshold)