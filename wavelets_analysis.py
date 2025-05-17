# XXX comentarios
# 1) ver bien los ejes y leyendas de los gráficos
# 2) ver que tengan sentido y revisar los nombres
# 3) hacer el plot del Señal denoised H_

import pandas as pd
import numpy as np
path = 'raw/AAP.N-2012-03-29-2024-03-01.csv'  # Asegúrate de que la ruta sea correcta
import matplotlib.pyplot as plt
from plots_code import plot_two_axis

# Paso 1: Cargar y preparar los datos
path = 'raw/A.N-2012-03-29-2024-03-01.csv'  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(path, index_col='Date', parse_dates=True)
df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
prices = df['log_return'].dropna() 

# Calcular precios acumulados para visualización
# Reconstruir la serie de precios a partir de log returns: P_t = P_0 * exp(cumsum(log_return))
initial_price = df['CLOSE'].iloc[0]
cumulative_log_returns = prices.cumsum()
cumulative_prices = initial_price * np.exp(cumulative_log_returns)
cumulative_prices = pd.Series(cumulative_prices, index=prices.index, name='Cumulative_Price')

# Ajustar la longitud a una potencia de 2 para simplificar el análisis wavelet
n = len(prices)
n_adjusted = 2**int(np.log2(n))
prices = prices[:n_adjusted]
cumulative_prices = cumulative_prices[:n_adjusted]
print(f"Longitud de la serie ajustada: {len(prices)}")

# Paso 2: Descomposición con Haar Wavelets hasta nivel 4
level = 4
coeffs = pywt.wavedec(prices.values, 'haar', level=level, mode='periodization') # coeffs = [cA4, cD4, cD3, cD2, cD1]
print("Longitudes de los coeficientes:")
print(f"cA4: {len(coeffs[0])}, cD4: {len(coeffs[1])}, cD3: {len(coeffs[2])}, cD2: {len(coeffs[3])}, cD1: {len(coeffs[4])}")

# Visualizar coeficientes wavelet
fig, axs = plt.subplots(len(coeffs), 1, figsize=(10, 10))
axs[0].plot(coeffs[0])
axs[0].set_title('Coeficientes de Aproximación (cA4)')
for i in range(1, len(coeffs)):
    axs[i].plot(coeffs[i])
    axs[i].set_title(f'Coeficientes de Detalle (cD{4 - i + 1})')
plt.tight_layout()
plt.savefig('plots/wavelet_coeffs.png')
plt.show()

# Paso 3: Calcular H_i(t) para i=1,...,4
H_series = {}
detail_reconstructed_series = {}  # Almacenar detail_reconstructed para cada nivel
for i in range(1, level + 1):  # i es el nivel de agregación para H
    details_sum = np.zeros_like(prices, dtype=float)
    detail_reconstructed_series[i] = {}
    for j in range(1, i + 1): # j es el índice del detalle D_j que se está reconstruyendo (D1, D2, ..., Di)
        # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
        # L = level (en tu caso, 4)
        # El coeficiente cD_j (ej: cD1, cD2) corresponde a coeffs[level - j + 1]
        
        idx_Dj_in_coeffs = level - j + 1 # Índice del cD_j actual en la lista 'coeffs'

        # Crear la lista de coeficientes para waverec para reconstruir solo D_j
        reconstruction_coeffs_list = []
        # 1. Coeficiente de aproximación cA_L (coeffs[0]) - siempre cero para reconstruir solo D_j
        reconstruction_coeffs_list.append(np.zeros_like(coeffs[0]))
        
        # 2. Coeficientes de detalle (cD_L, ..., cD_1)
        for k_coeffs_idx in range(1, len(coeffs)): # k_coeffs_idx va de 1 (para cD_L) a level (para cD_1)
            if k_coeffs_idx == idx_Dj_in_coeffs:
                reconstruction_coeffs_list.append(coeffs[k_coeffs_idx]) # Usar el cD_j actual
            else:
                reconstruction_coeffs_list.append(np.zeros_like(coeffs[k_coeffs_idx])) # Ceros para otros detalles
        
        print(f"Reconstruyendo D_{j}, longitudes de coeff_list: {[len(c) if hasattr(c, 'shape') else None for c in reconstruction_coeffs_list]}")
        try:
            detail_reconstructed = pywt.waverec(reconstruction_coeffs_list, 'haar', mode='periodization')
            detail_reconstructed = detail_reconstructed[:len(details_sum)] # Ajustar longitud
        except ValueError as e:
            print(f"Error en pywt.waverec para D_{j}: {e}")
            print(f"coeff_list estructura (longitudes): {[len(c) if hasattr(c, 'shape') else 'None o scalar' for c in reconstruction_coeffs_list]}")
            exit(1)
        details_sum += np.abs(detail_reconstructed) # sum if D_j
        detail_reconstructed_series[i][j] = detail_reconstructed
        
        detail_reconstructed_ser = pd.Series(detail_reconstructed,index=cumulative_prices.index)
        file_detalle = plot_two_axis(
                                    primary_series=cumulative_prices,
                                    secondary_series=detail_reconstructed_ser,
                                    primary_label="Precios Acumulados",
                                    secondary_label=f"Detalle Nivel {j}",
                                    primary_ylabel="Precio acumulado",
                                    secondary_ylabel=f"Detalle nivel {j}",
                                    title=f"Detalle Reconstruido Nivel {j} vs Precios Acumulados",
                                    filename=f"detail_reconstructed_level_{j}.png",
                                    secondary_color="red"       # opcional (por defecto ya es rojo)
                                )
        print(f"Gráfico guardado en: {file_detalle}")

    # Visualizar suma de detalles absolutos contra la serie de precios acumulados
    secondary_series = details_sum + cumulative_prices.mean()

    file_sum = plot_two_axis(
        primary_series=cumulative_prices,
        secondary_series=pd.Series(secondary_series,index=cumulative_prices.index),
        primary_label="Precios Acumulados",
        secondary_label=f"Suma Detalles hasta Nivel {i}",
        primary_ylabel="Precio acumulado",
        secondary_ylabel=f"Suma detalles nivel {i}",
        title=f"Suma de Detalles Absolutos hasta Nivel {i} vs Precios Acumulados",
        filename=f"details_sum_level_{i}.png",
        secondary_color="green"
    )
    print(f"Gráfico guardado en: {file_sum}")

    # Reconstruir aproximación A_L (usando cA_L = coeffs[0])
    coeff_list_approx_reconstruction = [coeffs[0]] # Usar el cA_L original
    for k_detail_idx in range(1, len(coeffs)): # Para todos los coeficientes de detalle cD_L, ..., cD_1
        coeff_list_approx_reconstruction.append(np.zeros_like(coeffs[k_detail_idx])) # Ceros para todos los detalles
    
    approx = pywt.waverec(coeff_list_approx_reconstruction, 'haar', mode='periodization')
    approx = approx[:len(details_sum)] # Ajustar longitud
    
    # Evitar división por cero
    approx = np.where(approx == 0, 1e-10, approx)
    H_series[i] = details_sum / approx
    print(f"H_{i} calculado, longitud: {len(H_series[i])}")

# Plot H_series
plt.figure(figsize=(12, 6))
for i in range(1, level + 1):
    plt.plot(prices.index, H_series[i], label=f'H_{i}')
plt.xlabel('Fecha')
plt.ylabel('H_i(t)')
plt.title('H_i(t) para cada nivel')
plt.legend()
plt.tight_layout()
plt.savefig('plots/H_series_all_levels.png')
plt.show()

# Paso 4: Calcular energía móvil (ventana de 30 días)
window_size = 30
energy_series = {}
for i in range(1, level + 1):
    energy_series[i] = pd.Series(H_series[i], index=prices.index).rolling(
        window=window_size, min_periods=1
    ).apply(lambda x: np.sum(x**2), raw=True).values
    print(f"Energía móvil H_{i}, longitud: {len(energy_series[i])}")

# Plot H_series
plt.figure(figsize=(12, 6))
for i in range(1, level + 1):
    plt.plot(prices.index, energy_series[i], label=f'H_{i}')
plt.xlabel('Fecha')
plt.ylabel('Energy_i(t)')
plt.title('Energy_i(t) para cada nivel')
plt.legend()
plt.tight_layout()
plt.savefig('plots/Energy_iall_levels.png')
# plt.show()

# Paso 5: Reducción de ruido (40% de energía retenida)
threshold_energy_retain = 0.4
denoised_series = {}
for i in range(1, level + 1):
    coeffs_H = pywt.wavedec(H_series[i], 'haar', level=level, mode='periodization')
    # Aplanar coeficientes, ignorando None
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

    cumulative_energy = np.cumsum(sorted_abs_coeffs**2)
    # Calcula la suma acumulada de las energías de los coeficientes (que están ordenados de mayor a menor).
    # El primer elemento es la energía del coeficiente más grande.
    # El segundo elemento es la suma de las energías de los dos coeficientes más grandes, y así sucesivamente.
    # El último elemento de cumulative_energy es igual a total_energy.

    j = np.searchsorted(cumulative_energy, threshold_energy_retain * total_energy)
    # 1. 0.4 * total_energy: Calcula el 40% de la energía total. Este es el objetivo de energía a retener.
    # 2. np.searchsorted(cumulative_energy, ...): Encuentra el índice 'j' en el array 'cumulative_energy'
    #    tal que todos los elementos hasta cumulative_energy[j-1] suman menos del 40% de la energía total,
    #    y al incluir cumulative_energy[j], se alcanza o supera ese 40%.
    #    En otras palabras, los 'j' coeficientes más grandes (según sorted_abs_coeffs) contienen
    #    aproximadamente el 40% de la energía total de la señal H_series[i].

    T = sorted_abs_coeffs[j] if j < len(sorted_abs_coeffs) else sorted_abs_coeffs[-1]
    # Establece el valor del umbral 'T'.
    # T es el valor absoluto del j-ésimo coeficiente más grande (es decir, el coeficiente más pequeño
    # entre los que contribuyen al 40% de la energía retenida).
    # Los coeficientes cuya magnitud sea menor que T serán considerados "ruido".
    # La condición 'if j < len(sorted_abs_coeffs)' maneja el caso límite donde 'j' podría
    # estar fuera de los límites del array (por ejemplo, si se retiene el 100% de la energía).

    # Umbral duro
    thresholded_coeffs = [pywt.threshold(c, T, mode='hard') if c is not None else c for c in coeffs_H]
    # Aplica la técnica de "umbral duro" a cada conjunto de coeficientes en la lista original coeffs_H
    # (que eran [cA_H, cD_L_H, ..., cD_1_H]).
    # Para cada coeficiente en cada array 'c' de coeffs_H:
    #  - Si la magnitud del coeficiente (abs(coef)) es mayor que T, el coeficiente se mantiene sin cambios.
    #  - Si la magnitud del coeficiente (abs(coef)) es menor o igual a T, el coeficiente se establece en 0.
    # Esto elimina los coeficientes considerados "ruido".

    denoised_series[i] = pywt.waverec(thresholded_coeffs, 'haar', mode='periodization')[:len(H_series[i])]
    # Reconstruye la señal H_series[i] utilizando los coeficientes wavelet umbralizados (thresholded_coeffs).
    # El resultado, denoised_series[i], es una versión con ruido reducido (denoised) de la H_series[i] original.
    # Se espera que esta serie "limpia" revele de forma más clara los picos o características significativas.
    # [:len(H_series[i])] asegura que la serie reconstruida tenga la misma longitud que la H_series[i] original.

    print(f"Señal denoised H_{i}, longitud: {len(denoised_series[i])}")

# Paso 6: Identificar fechas críticas y señales de alerta
critical_dates = {}
for i in range(1, level + 1):
    critical_indices = np.where(np.abs(denoised_series[i]) > 1e-10)[0]  # Tolerancia para ceros
    critical_dates[i] = prices.index[critical_indices]
    print(f"H_{i} - Fechas críticas: {len(critical_dates[i])}")

# --- Ejemplo para graficar H_i y su denoised_series[i]
level_to_plot_denoising = 1 # Elige un nivel, por ejemplo H_1

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(prices.index, H_series[level_to_plot_denoising], label=f'H_{level_to_plot_denoising} (Observada)')
plt.title(f'H_{level_to_plot_denoising} (Observada)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(prices.index, denoised_series[level_to_plot_denoising], label=f'Denoised H_{level_to_plot_denoising}', color='orange')
plt.title(f'Denoised H_{level_to_plot_denoising} (40% Energía Retenida)')
plt.legend()

plt.tight_layout()
plt.savefig(f'plots/H{level_to_plot_denoising}_vs_denoised.png')
plt.show()
print(f"Gráfico de H_{level_to_plot_denoising} y su versión denoised guardado.")


# --- Plot Denoised H_series
plt.figure(figsize=(12, 6))
for i in range(1, level + 1):
    # Asegurarse de que el índice de tiempo se alinee si es necesario (prices.index debería funcionar)
    plt.plot(prices.index[:len(denoised_series[i])], denoised_series[i], label=f'Denoised H_{i}')
plt.xlabel('Fecha')
plt.ylabel('Denoised H_i(t)')
plt.title('Denoised H_i(t) para cada nivel (40% Energía Retenida)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/denoised_H_series_all_levels.png')
plt.show()
print("Gráfico de todas las Denoised H_series guardado.")


# Sistema de contador para señales de alerta
early_warnings = {}
for i in range(1, level + 1): # i se refiere al H_i que se está analizando
    early_warnings[i] = []
    for crit_date in critical_dates[i]:
        crit_idx = prices.index.get_loc(crit_date)
        counter = 0
        for x in range(max(0, crit_idx - 15), min(len(prices), crit_idx + 16)):
            # Crear serie modificada
            truncated = prices.iloc[:x+1].values
            flipped = truncated[::-1]
            modified = np.concatenate([truncated, flipped] * (len(prices) // (2 * (x+1)) + 1))[:len(prices)]
            coeffs_mod = pywt.wavedec(modified, 'haar', level=level, mode='periodization')
            details_sum_mod = np.zeros_like(prices, dtype=float)
            
            # El bucle 'j' aquí se refiere a la suma de los D_j para construir H_i
            # (el mismo 'j' que en el cálculo original de H_series)
            for j in range(1, i + 1): # Suma de |D_1| hasta |D_i|
                # Índice en coeffs_mod para el cD_j que se está reconstruyendo
                idx_Dj_in_coeffs_mod = level - j + 1 

                coeff_list_detail_mod_reconstruct = []
                # 1. Coeficiente de aproximación cA_mod (coeffs_mod[0]) - siempre cero
                coeff_list_detail_mod_reconstruct.append(np.zeros_like(coeffs_mod[0]))
                
                # 2. Coeficientes de detalle cD_L_mod, ..., cD_1_mod
                for k_idx_mod in range(1, len(coeffs_mod)): # k_idx_mod de 1 a level
                    if k_idx_mod == idx_Dj_in_coeffs_mod:
                        coeff_list_detail_mod_reconstruct.append(coeffs_mod[k_idx_mod])
                    else:
                        coeff_list_detail_mod_reconstruct.append(np.zeros_like(coeffs_mod[k_idx_mod]))
                
                detail_reconstructed = pywt.waverec(coeff_list_detail_mod_reconstruct, 'haar', mode='periodization')[:len(details_sum_mod)]
                details_sum_mod += np.abs(detail_reconstructed)
            
            # Reconstruir aproximación A_L_mod (usando cA_L_mod = coeffs_mod[0])
            coeff_list_approx_mod_reconstruct = [coeffs_mod[0]] # cA_L_mod
            for k_idx_mod_detail in range(1, len(coeffs_mod)):
                coeff_list_approx_mod_reconstruct.append(np.zeros_like(coeffs_mod[k_idx_mod_detail]))
            
            approx_mod = pywt.waverec(coeff_list_approx_mod_reconstruct, 'haar', mode='periodization')[:len(details_sum_mod)]
            approx_mod = np.where(approx_mod == 0, 1e-10, approx_mod)
            H_mod = details_sum_mod / approx_mod
            
            # La parte de thresholded_coeffs_mod y pywt.waverec para denoised_mod ya debería estar bien
            # siempre que coeffs_H_mod (resultado de wavedec) no contenga Nones.
            coeffs_H_mod = pywt.wavedec(H_mod, 'haar', level=level, mode='periodization')
            flat_coeffs_mod = np.concatenate([c for c in coeffs_H_mod if c is not None]) # c is not None es una buena guarda pero wavedec no debería dar Nones
            sorted_abs_coeffs_mod = np.sort(np.abs(flat_coeffs_mod))[::-1]
            cumulative_energy_mod = np.cumsum(sorted_abs_coeffs_mod**2)
            j_mod = np.searchsorted(cumulative_energy_mod, 0.4 * total_energy)
            T_mod = sorted_abs_coeffs_mod[j_mod] if j_mod < len(sorted_abs_coeffs_mod) else sorted_abs_coeffs_mod[-1]
            
            thresholded_coeffs_mod = [pywt.threshold(c, T_mod, mode='hard') for c in coeffs_H_mod] # Asumiendo que c nunca es None
            denoised_mod = pywt.waverec(thresholded_coeffs_mod, 'haar', mode='periodization')[:len(H_mod)]

            if crit_idx < len(denoised_mod) and np.abs(denoised_mod[crit_idx]) > 1e-10:
                counter += 1
        if counter > 10:
            early_warnings[i].append(crit_date)
    print(f"H_{i} - Señales de alerta calculadas: {len(early_warnings[i])}")


# Resultados
for i in range(1, level + 1):
    print(f"H_{i} - Señales de alerta temprana:")
    if early_warnings[i]:
        for date in early_warnings[i]:
            print(f"  {date.strftime('%Y-%m-%d')}")
    else:
        print("  Ninguna señal detectada.")