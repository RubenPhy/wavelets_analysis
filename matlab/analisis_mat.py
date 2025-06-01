from   scipy.io   import  loadmat
import numpy      as      np
matrixVar = loadmat( "matlab/Date.mat" )

# Do whatever data manipulation you need here

# Do more stuff here if needed
# ...
print( 'Done processing' )

import numpy as np
import pywt


def _reconstruir_coeficiente(coef_list, tipo, nivel, wavelet, longitud_original):
    """
    Replica la función MATLAB `wrcoef`.
    Devuelve la reconstrucción (aproximación 'a' o detalle 'd')
    del nivel solicitado, reescalada a la longitud original.
    """
    # Copiamos la lista de coeficientes y ponemos a cero
    # todos los que no tocan
    coefs = [c.copy() for c in coef_list]
    if tipo == 'a':
        # Cero en todos los detalles
        for j in range(1, len(coefs)):
            coefs[j] = np.zeros_like(coefs[j])
    elif tipo == 'd':
        # Cero en la aproximación y en TODOS los detalles ≠ nivel
        coefs[0] = np.zeros_like(coefs[0])
        for j in range(1, len(coefs)):
            if j != nivel:
                coefs[j] = np.zeros_like(coefs[j])
    else:
        raise ValueError("tipo debe ser 'a' o 'd'")

    # Reconstruimos
    rec = pywt.waverec(coefs, wavelet)
    # Ajustamos longitud (waverec a veces devuelve 1-2 muestras extra)
    return rec[:longitud_original]


def detcrash(signal, level, window):
    """
    Traducción Python de la función MATLAB `detcrash`.

    Parameters
    ----------
    signal : array-like
        Señal de entrada (1 D).
    level : int
        Profundidad de la descomposición wavelet (>=1).
    window : int
        Semiancho de la ventana simétrica (tamaño total 2*window).

    Returns
    -------
    variabsig : np.ndarray
        Variabilidad normalizada en cada instante.
    """
    signal = np.asarray(signal, dtype=float)
    le = len(signal)

    variabsig = np.zeros(le)

    # ------------------------------------------------------------------
    #  Primero simetrizamos la señal alrededor de cada instante,
    #  con longitud total 2*window
    # ------------------------------------------------------------------
    for i in range(window - 1, le):

        s1 = signal[i - window + 1:i + 1]           # Ventana hacia atrás
        s2 = s1[::-1]                               # Parte reflejada
        s = np.concatenate((s1, s2))                # Señal simétrica

        # --------------------------------------------------------------
        #  Descomposición wavelet Haar hasta 'level'
        # --------------------------------------------------------------
        C = pywt.wavedec(s, 'haar', level=level)

        # --------------------------------------------------------------
        #  Cálculo de detalles reconstruidos a todos los niveles
        # --------------------------------------------------------------
        detalles_rec = []
        for j in range(1, level + 1):
            # Nota: en PyWavelets, el índice 1 es el detalle de máximo nivel
            # MATLAB usa índice 1 = primer nivel de detalle
            detalle_j = _reconstruir_coeficiente(C, 'd', j, 'haar', len(s))
            detalles_rec.append(detalle_j)
        detalles_rec = np.stack(detalles_rec, axis=1)  # ↔ d(:,j) MATLAB

        # --------------------------------------------------------------
        #  Cálculo de la aproximación al último nivel
        # --------------------------------------------------------------
        ap = _reconstruir_coeficiente(C, 'a', 0, 'haar', len(s))

        # --------------------------------------------------------------
        #  Variabilidad de los detalles al último nivel
        # --------------------------------------------------------------
        det_last = detalles_rec[:, level - 1]          # Nivel 'level'
        difdet = np.abs(np.diff(det_last))             # |d(k+1) - d(k)|
        varydet = np.sum(difdet)

        # --------------------------------------------------------------
        #  Normalización de la variabilidad
        # --------------------------------------------------------------
        varydetnorm = varydet / np.sum(ap)

        # Posicionamos el valor normalizado en el instante i (0-index)
        variabsig[i] = varydetnorm

    return variabsig
