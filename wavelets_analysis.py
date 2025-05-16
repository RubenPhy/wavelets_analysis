import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Paso 1: Cargar y preparar los datos
path = 'raw/A.N-2012-03-29-2024-03-01.csv'  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(path, index_col='Date', parse_dates=True)
df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
prices = df['log_return'].dropna() 

# Ajustar la longitud a una potencia de 2 para simplificar el análisis wavelet
n = len(prices)
n_adjusted = 2**int(np.log2(n))
prices = prices[:n_adjusted]
print(f"Longitud de la serie ajustada: {len(prices)}")

# Paso 2: Descomposición con Haar Wavelets hasta nivel 4
level = 4
coeffs = pywt.wavedec(prices.values, 'haar', level=level, mode='periodization')
# coeffs = [cA4, cD4, cD3, cD2, cD1]
print("Longitudes de los coeficientes:")
print(f"cA4: {len(coeffs[0])}, cD4: {len(coeffs[1])}, cD3: {len(coeffs[2])}, cD2: {len(coeffs[3])}, cD1: {len(coeffs[4])}")

# Paso 3: Calcular H_i(t) para i=1,...,4
H_series = {}
for i in range(1, level + 1):  # i es el nivel de agregación para H
    details_sum = np.zeros_like(prices, dtype=float)
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
        details_sum += np.abs(detail_reconstructed)
        
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


# Paso 4: Calcular energía móvil (ventana de 30 días)
window_size = 30
energy_series = {}
for i in range(1, level + 1):
    energy_series[i] = pd.Series(H_series[i], index=prices.index).rolling(
        window=window_size, min_periods=1
    ).apply(lambda x: np.sum(x**2), raw=True).values
    print(f"Energía móvil H_{i}, longitud: {len(energy_series[i])}")

# Paso 5: Reducción de ruido (40% de energía retenida)
denoised_series = {}
for i in range(1, level + 1):
    coeffs_H = pywt.wavedec(H_series[i], 'haar', level=level, mode='periodization')
    # Aplanar coeficientes, ignorando None
    flat_coeffs = np.concatenate([c for c in coeffs_H if c is not None])
    sorted_abs_coeffs = np.sort(np.abs(flat_coeffs))[::-1]
    total_energy = np.sum(sorted_abs_coeffs**2)
    cumulative_energy = np.cumsum(sorted_abs_coeffs**2)
    j = np.searchsorted(cumulative_energy, 0.4 * total_energy)
    T = sorted_abs_coeffs[j] if j < len(sorted_abs_coeffs) else sorted_abs_coeffs[-1]
    # Umbral duro
    thresholded_coeffs = [pywt.threshold(c, T, mode='hard') if c is not None else c for c in coeffs_H]
    denoised_series[i] = pywt.waverec(thresholded_coeffs, 'haar', mode='periodization')[:len(H_series[i])]
    print(f"Señal denoised H_{i}, longitud: {len(denoised_series[i])}")

# Paso 6: Identificar fechas críticas y señales de alerta
critical_dates = {}
for i in range(1, level + 1):
    critical_indices = np.where(np.abs(denoised_series[i]) > 1e-10)[0]  # Tolerancia para ceros
    critical_dates[i] = prices.index[critical_indices]
    print(f"H_{i} - Fechas críticas: {len(critical_dates[i])}")

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