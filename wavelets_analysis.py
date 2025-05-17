import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from plots_code import plot_two_axis

# --- XXX Comentarios Generales ---
# 1) Ejes y leyendas: Se ha intentado que sean descriptivos.
# 2) Sentido de los gráficos y nombres: Revisados para mayor claridad.
# 3) Plot de Señal Denoised H_i: Añadido.
# ---------------------------------

# Paso 1: Cargar y preparar los datos
path = 'raw/AAP.N-2012-03-29-2024-03-01.csv'
df = pd.read_csv(path, index_col='Date', parse_dates=True)
df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
prices = df['log_return'].dropna() 

# Calcular precios acumulados para visualización
initial_price = df['CLOSE'].dropna().iloc[0]
cumulative_log_returns = prices.cumsum()
cumulative_prices = initial_price * np.exp(cumulative_log_returns)
cumulative_prices = pd.Series(cumulative_prices, index=prices.index, name='Cumulative_Price')

# Ajustar la longitud a una potencia de 2 para simplificar el análisis wavelet
n = len(prices)
n_adjusted = 2**int(np.log2(n))
prices = prices.iloc[:n_adjusted]
cumulative_prices = cumulative_prices.iloc[:n_adjusted]
print(f"Longitud de la serie ajustada: {len(prices)}")

# Paso 2: Descomposición con Haar Wavelets hasta nivel 4
level = 4
coeffs = pywt.wavedec(prices.values, 'haar', level=level, mode='periodization') # coeffs = [cA_L, cD_L, ..., cD_1]
print("Longitudes de los coeficientes:")
# Los coeficientes son [cA_level, cD_level, cD_{level-1}, ..., cD_1]
# Por lo tanto, coeffs[0] es cA4, coeffs[1] es cD4, coeffs[2] es cD3, coeffs[3] es cD2, coeffs[4] es cD1.
print(f"cA{level}: {len(coeffs[0])}")
for i_coeff in range(1, len(coeffs)):
    print(f"cD{level - i_coeff + 1}: {len(coeffs[i_coeff])}")

# Visualizar coeficientes wavelet
fig, axs = plt.subplots(len(coeffs), 1, figsize=(12, 10), sharex=False)
axs[0].plot(coeffs[0], label=f'cA{level}')
axs[0].set_title(f'Coeficientes de Aproximación (cA{level})')
axs[0].legend(loc='upper right')
for i_coeff in range(1, len(coeffs)):
    axs[i_coeff].plot(coeffs[i_coeff], label=f'cD{level - i_coeff + 1}')
    axs[i_coeff].set_title(f'Coeficientes de Detalle (cD{level - i_coeff + 1})')
    axs[i_coeff].legend(loc='upper right')
fig.suptitle('Coeficientes Wavelet de los Retornos Logarítmicos', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el título general
plt.savefig('plots/wavelet_coefficients_log_returns.png')
print("Gráfico de coeficientes wavelet guardado en: plots/wavelet_coefficients_log_returns.png")
plt.show()

# Paso 3: Calcular H_i(t) para i=1,...,level
H_series = {}
detail_reconstructed_series = {}  # Almacenar detail_reconstructed para cada nivel
print("\n--- Iniciando Cálculo de H_i(t) y Reconstrucción de Detalles ---")
for i_h_level in range(1, level + 1): # i_h_level es el 'i' en H_i (nivel de agregación para H)
    details_sum = np.zeros_like(prices.values, dtype=float) # Usar .values para asegurar array numpy
    detail_reconstructed_series[i_h_level] = {}
    print(f"\nCalculando para H_{i_h_level}:")
    for j_detail_level in range(1, i_h_level + 1): # j_detail_level es el 'j' en D_j (nivel del detalle individual)
        # El coeficiente cD_j (ej: D1, D2) corresponde a coeffs[level - j_detail_level + 1]
        idx_Dj_in_coeffs = level - j_detail_level + 1

        reconstruction_coeffs_list = [np.zeros_like(coeffs[0])] # cA_L = 0
        for k_coeffs_idx in range(1, len(coeffs)):
            if k_coeffs_idx == idx_Dj_in_coeffs:
                reconstruction_coeffs_list.append(coeffs[k_coeffs_idx])
            else:
                reconstruction_coeffs_list.append(np.zeros_like(coeffs[k_coeffs_idx]))
        
        # print(f"  Reconstruyendo D_{j_detail_level} para H_{i_h_level}, longitudes de coeff_list: {[len(c) if hasattr(c, 'shape') else None for c in reconstruction_coeffs_list]}")
        try:
            detail_reconstructed = pywt.waverec(reconstruction_coeffs_list, 'haar', mode='periodization')
            detail_reconstructed = detail_reconstructed[:len(details_sum)]
        except ValueError as e:
            print(f"  Error en pywt.waverec para D_{j_detail_level}: {e}")
            exit(1)
        
        details_sum += np.abs(detail_reconstructed)
        detail_reconstructed_series[i_h_level][j_detail_level] = detail_reconstructed
        
        # Visualizar detalle reconstruido individual D_j vs Precios Acumulados (usando plot_two_axis)
        detail_reconstructed_pd_series = pd.Series(detail_reconstructed, index=cumulative_prices.index, name=f"Detalle_D{j_detail_level}")
        file_detalle_path = plot_two_axis(
            primary_series=cumulative_prices,
            secondary_series=detail_reconstructed_pd_series,
            primary_label="Precios Acumulados",
            secondary_label=f"Detalle D_{j_detail_level} (de Retornos Log.)",
            primary_ylabel="Precio Acumulado",
            secondary_ylabel=f"Amplitud Detalle D_{j_detail_level}",
            title=f"Precio Acumulado vs. Detalle D_{j_detail_level} Reconstruido",
            filename=f"detail_D{j_detail_level}_reconstructed_for_H{i_h_level}.png", # Nombre más específico
            secondary_color="red"
        )
        print(f"  Gráfico de D_{j_detail_level} guardado en: {file_detalle_path}")

    # Visualizar suma de detalles absolutos (para H_i) vs Precios Acumulados
    # Lo que se grafica es details_sum (de retornos) desplazado por la media de los precios, en un segundo eje.
    details_sum_pd_series = pd.Series(details_sum, index=cumulative_prices.index, name=f"SumaAbsDetalles_H{i_h_level}")
    file_sum_path = plot_two_axis(
        primary_series=cumulative_prices,
        # secondary_series=pd.Series(details_sum + cumulative_prices.mean(), index=cumulative_prices.index), # Esto mezcla escalas
        secondary_series=details_sum_pd_series, # Graficar la suma de detalles directamente
        primary_label="Precios Acumulados",
        secondary_label=f"Suma |D_j| hasta j={i_h_level} (para H_{i_h_level})",
        primary_ylabel="Precio Acumulado",
        secondary_ylabel=f"Suma Absoluta Detalles",
        title=f"Precio Acumulado vs. Suma Detalles Absolutos para H_{i_h_level}",
        filename=f"sum_abs_details_for_H{i_h_level}.png",
        secondary_color="green"
    )
    print(f"  Gráfico de Suma Detalles para H_{i_h_level} guardado en: {file_sum_path}")

    # Reconstruir aproximación A_L (usando cA_L = coeffs[0])
    coeff_list_approx_reconstruction = [coeffs[0]] # Usar el cA_L original
    for k_detail_idx in range(1, len(coeffs)): # Para todos los coeficientes de detalle cD_L, ..., cD_1
        coeff_list_approx_reconstruction.append(np.zeros_like(coeffs[k_detail_idx])) # Ceros para todos los detalles
    
    approx = pywt.waverec(coeff_list_approx_reconstruction, 'haar', mode='periodization')
    approx = approx[:len(details_sum)] # Ajustar longitud
    
    # Evitar división por cero
    approx = np.where(approx == 0, 1e-10, approx)
    H_series[i_h_level] = details_sum / approx
    print(f"  H_{i_h_level} calculado, longitud: {len(H_series[i_h_level])}")

# Plot H_series todas juntas
plt.figure(figsize=(12, 6))
for i_h_level_plot in range(1, level + 1):
    plt.plot(prices.index, H_series[i_h_level_plot], label=f'$H_{i_h_level_plot}(t)$')
plt.xlabel('Fecha')
plt.ylabel('$H_i(t)$')
plt.title('$H_i(t)$ para cada Nivel de Agregación de Detalles')
plt.legend()
plt.tight_layout()
plt.savefig('plots/H_series_all_levels.png')
print("\nGráfico de todas las H_series guardado en: plots/H_series_all_levels.png")
plt.show()


# Paso 4: Calcular energía móvil (ventana de 30 días)
print("\n--- Iniciando Cálculo de Energía Móvil ---")
window_size = 30
energy_series = {}
for i_h_level in range(1, level + 1):
    # Asegurarse que H_series[i_h_level] es un array 1D para pd.Series
    h_series_array = np.asarray(H_series[i_h_level])
    energy_series[i_h_level] = pd.Series(h_series_array, index=prices.index).rolling(
        window=window_size, min_periods=1
    ).apply(lambda x: np.sum(x**2), raw=True).values
    print(f"Energía móvil H_{i_h_level}, longitud: {len(energy_series[i_h_level])}")

# Plot Energy_series todas juntas
plt.figure(figsize=(12, 6))
for i_h_level_plot in range(1, level + 1):
    plt.plot(prices.index, energy_series[i_h_level_plot], label=f'$Energy(H_{i_h_level_plot})$')
plt.xlabel('Fecha')
plt.ylabel('Energía Móvil de $H_i(t)$')
plt.title('Energía Móvil de $H_i(t)$ para cada Nivel (Ventana 30 días)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/Energy_series_all_levels.png')
print("Gráfico de todas las Energy_series guardado en: plots/Energy_series_all_levels.png")
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
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(prices.index, H_series[level_to_plot_denoising], label=f'$H_{level_to_plot_denoising}$ (Observada)')
    plt.title(f'$H_{level_to_plot_denoising}(t)$ (Observada)')
    plt.ylabel('Amplitud')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(prices.index, denoised_series[level_to_plot_denoising], label=f'Denoised $H_{level_to_plot_denoising}$', color='orange')
    plt.title(f'Denoised $H_{level_to_plot_denoising}(t)$ ({threshold_energy_retain*100}% Energía Retenida)')
    plt.xlabel('Fecha')
    plt.ylabel('Amplitud')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/H{level_to_plot_denoising}_vs_denoised.png')
    print(f"\nGráfico de H_{level_to_plot_denoising} y su versión denoised guardado en: plots/H{level_to_plot_denoising}_vs_denoised.png")
    plt.show()

# --- Plot de todas las Denoised H_series (similar a p.24 del PDF) ---
plt.figure(figsize=(12, 6))
for i_plot in range(1, level + 1):
    if i_plot in denoised_series:
        plt.plot(prices.index[:len(denoised_series[i_plot])], denoised_series[i_plot], label=f'Denoised $H_{i_plot}$')
plt.xlabel('Fecha')
plt.ylabel('Amplitud Denoised $H_i(t)$')
plt.title(f'Denoised $H_i(t)$ para cada Nivel ({threshold_energy_retain*100}% Energía Retenida)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/denoised_H_series_all_levels.png')
print("Gráfico de todas las Denoised H_series guardado en: plots/denoised_H_series_all_levels.png")
plt.show()


# Paso 6: Identificar fechas críticas y Sistema de contador para señales de alerta
print("\n--- Iniciando Identificación de Fechas Críticas y Sistema de Alerta ---")
critical_dates = {}
for i_h_level in range(1, level + 1):
    critical_indices = np.where(np.abs(denoised_series[i_h_level]) > 1e-10)[0] # Tolerancia para ceros
    critical_dates[i_h_level] = prices.index[critical_indices]
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
            original_series_idx = prices.index.get_loc(crit_date)
            counter = 0 # Inicializa un contador para esta fecha crítica específica.

            # Este bucle simula "terminar" la serie de precios en diferentes puntos 'x'
            # alrededor de la fecha crítica 'crit_idx'.
            # Se define una ventana de 31 días: 15 días antes de crit_idx, crit_idx mismo, y 15 días después.
            # max(0, crit_idx - 15): asegura que no empecemos antes del inicio de la serie.
            # min(len(prices), crit_idx + 16): asegura que no nos pasemos del final de la serie. 
            # Bucle de simulación para robustez
            for x_sim_idx in range(max(0, original_series_idx - 15), min(len(prices), original_series_idx + 16)):
                truncated = prices.iloc[:x_sim_idx+1].values
                if not truncated.size: continue # Saltar si truncado está vacío
                # -- Inicio de la Simulación para el punto 'x' --
                # El objetivo es ver si la fecha 'crit_date' todavía se detectaría como
                # anómala si solo hubiéramos tenido datos hasta el día 'x'.

                # Crear serie modificada
                # truncated: Toma la serie de retornos logarítmicos ('prices') desde el inicio hasta el día 'x'.                
                flipped = truncated[::-1]
                # modified: Crea una nueva serie de longitud igual a la 'prices' original.
                # Lo hace concatenando la 'truncated' y su 'flipped' repetidamente.
                # Esto es una técnica para extender artificialmente la serie truncada para el análisis wavelet,
                # tratando de minimizar los efectos de borde y mantener cierta estructura estadística.
                # (len(prices) // (2 * (x+1)) + 1): Calcula cuántas veces se necesita repetir el patrón [truncated, flipped].

                # Asegurar que el patrón para concatenar no sea vacío
                pattern_len = 2 * len(truncated)
                if pattern_len == 0: continue

                num_repeats = (len(prices) // pattern_len) + 1
                modified = np.concatenate([truncated, flipped] * num_repeats)[:len(prices)]
                
                coeffs_mod = pywt.wavedec(modified, 'haar', level=level, mode='periodization')
                details_sum_mod = np.zeros_like(prices.values, dtype=float)
                
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
plt.savefig(f'plots/cumulative_prices_with_H{level_to_plot_warnings}_warnings.png')
plt.show()
print(f"Gráfico de precios con alertas H_{level_to_plot_warnings} guardado en: plots/cumulative_prices_with_H{level_to_plot_warnings}_warnings.png")


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
# Filtrar crisis que caen dentro del rango de fechas de 'prices'
valid_crisis_dates = {
    name: dt for name, dt in crisis_dates_in_data.items() 
    if prices.index.min() <= dt <= prices.index.max()
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
plt.savefig('plots/frequency_critical_dates_robustness_check.png', bbox_inches='tight')
print("\nGráfico de frecuencia de robustez de fechas críticas guardado en: plots/frequency_critical_dates_robustness_check.png")
plt.show()

print("\n--- Análisis Completado ---")