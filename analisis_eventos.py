import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from wavelets_analysis import extract_extreme_dates  # Importamos la función necesaria

events = {
    'Consumer Discretionary': {
        '2015-08-24': 'Corrección del mercado debido a preocupaciones sobre la economía china.',
        '2015-09-18': 'Escándalo de emisiones de Volkswagen, afectando acciones automotrices.',
        '2017-09-18': 'Toys "R" Us declara bancarrota, impactando el sector retail.',
        '2018-10-15': 'Sears declara bancarrota, afectando el comercio minorista.',
        '2018-12-24': 'S&P 500 cierra en su nivel más bajo del año por tensiones comerciales y subidas de tasas.',
        '2020-03-11': 'OMS declara la pandemia de COVID-19, causando una caída en el gasto discrecional.',
        '2020-05-04': 'J.Crew declara bancarrota durante la pandemia.'
    },
    'Consumer Staples': {
        '2015-08-24': 'Corrección del mercado afectando todos los sectores.',
        '2018-12-24': 'Venta masiva en el mercado impactando precios de acciones.',
        '2019-02-22': 'Acciones de Kraft Heinz caen por malos resultados y investigación de la SEC.',
        '2020-03-11': 'Pandemia de COVID-19 causa volatilidad inicial antes de estabilizarse en bienes esenciales.',
        '2016-04-20': 'Brote de listeria en Blue Bell Creameries lleva a un recall, afectando acciones.',
        '2022-01-01': 'Preocupaciones por inflación comienzan a impactar a empresas de bienes de consumo.',  # Fecha aproximada
        '2018-01-01': 'Volatilidad general del mercado afectando menos a este sector defensivo.'  # Fecha aproximada
    },
    'Energy': {
        '2014-11-27': 'OPEP decide no reducir producción, causando una caída en los precios del petróleo.',
        '2015-08-24': 'Precios del petróleo caen por preocupaciones sobre la economía china.',
        '2016-01-20': 'Precios del petróleo alcanzan un mínimo de 12 años.',
        '2020-03-09': 'Precios del petróleo colapsan tras la guerra de precios Rusia-Arabia Saudita.',
        '2020-04-20': 'Precios del WTI caen a negativo por primera vez en la historia.',
        '2022-02-24': 'Invasión de Rusia a Ucrania genera volatilidad en el mercado energético.',
        '2023-01-01': 'Evento hipotético: Nuevas políticas energéticas impactan el sector.'  # Fecha aproximada
    },
    'Financials': {
        '2012-05-18': 'IPO de Facebook enfrenta problemas, afectando acciones financieras y tecnológicas.',
        '2013-10-01': 'Inicio del cierre del gobierno de EE.UU., causando incertidumbre en el mercado.',
        '2015-08-24': 'Corrección del mercado impactando acciones financieras.',
        '2016-06-24': 'Resultados del voto del Brexit generan volatilidad global.',
        '2018-12-24': 'Venta masiva por subidas de tasas y tensiones comerciales.',
        '2020-03-11': 'Pandemia de COVID-19 impacta severamente los mercados financieros.',
        '2023-03-10': 'Colapso de Silicon Valley Bank sacude el sector financiero.'
    },
    'Health Care': {
        '2012-06-28': 'Corte Suprema ratifica la Ley de Cuidado de Salud Asequible, afectando seguros de salud.',
        '2015-09-21': 'Controversia por aumento de precios de Turing Pharmaceuticals.',
        '2016-08-22': 'Escándalo de precios de EpiPen por Mylan impacta acciones farmacéuticas.',
        '2017-07-28': 'Senado vota en contra de derogar Obamacare, estabilizando el sector.',
        '2020-03-11': 'Pandemia de COVID-19 causa caídas iniciales antes de aumentar la demanda sanitaria.',
        '2019-01-01': 'Litigios por la crisis de opioides afectan a empresas farmacéuticas.',  # Fecha aproximada
        '2021-01-01': 'Desafíos en la distribución de vacunas impactan acciones de salud.'  # Fecha aproximada
    },
    'Industrials': {
        '2018-07-06': 'EE.UU. impone aranceles a bienes chinos, afectando manufactura e industrias.',
        '2020-03-11': 'Pandemia de COVID-19 interrumpe cadenas de suministro y producción industrial.',
        '2022-02-24': 'Conflicto Rusia-Ucrania afecta el comercio global e industrial.',
        '2015-08-24': 'Corrección del mercado por preocupaciones económicas globales.',
        '2016-06-24': 'Voto del Brexit genera incertidumbre en inversiones industriales.',
        '2019-05-10': 'Escalada de la guerra comercial EE.UU.-China con nuevos aranceles.',
        '2023-01-01': 'Evento hipotético: Nuevas regulaciones industriales o políticas económicas.'  # Fecha aproximada
    },
    'Information Technology': {
        '2018-03-17': 'Escándalo de Cambridge Analytica en Facebook afecta acciones tecnológicas.',
        '2020-03-11': 'Pandemia de COVID-19 causa caídas iniciales en tecnología antes de recuperación.',
        '2022-01-01': 'Inicio de venta masiva en tecnología por subidas de tasas y preocupaciones de valoración.',  # Fecha aproximada
        '2015-08-24': 'Corrección del mercado impacta el sector tecnológico.',
        '2016-06-24': 'Voto del Brexit genera volatilidad en inversiones tecnológicas.',
        '2019-05-13': 'EE.UU. prohíbe a Huawei, afectando cadenas de suministro tecnológicas.',
        '2023-01-01': 'Evento hipotético: Acciones antimonopolio contra grandes tecnológicas.'  # Fecha aproximada
    },
    'Materials': {
        '2015-08-24': 'Colapso del mercado chino afecta precios de commodities y el sector de materiales.',
        '2020-03-11': 'Pandemia de COVID-19 reduce la demanda de materiales industriales.',
        '2022-02-24': 'Conflicto Rusia-Ucrania interrumpe cadenas de suministro de materias primas.',
        '2018-07-06': 'Guerra comercial EE.UU.-China impacta exportaciones de materiales.',
        '2016-01-20': 'Precios de commodities alcanzan mínimos por desaceleración económica global.',
        '2019-05-10': 'Escalada de tensiones comerciales afecta el sector de materiales.',
        '2023-01-01': 'Evento hipotético: Nuevas regulaciones ambientales impactan producción de materiales.'  # Fecha aproximada
    },
    'Real Estate': {
        '2018-12-19': 'La Reserva Federal sube tasas de interés, afectando tasas hipotecarias y acciones inmobiliarias.',
        '2020-03-11': 'Pandemia de COVID-19 impacta severamente el real estate comercial.',
        '2022-03-16': 'Inicio de subidas de tasas por la Fed, aumentando costos de préstamo para bienes raíces.',
        '2015-08-24': 'Corrección del mercado afecta inversiones inmobiliarias.',
        '2016-06-24': 'Voto del Brexit genera incertidumbre en mercados inmobiliarios globales.',
        '2019-01-01': 'Desaceleración del mercado de vivienda por problemas de asequibilidad.',  # Fecha aproximada
        '2023-01-01': 'Evento hipotético: Cambios en leyes de impuestos o regulaciones de zonificación.'  # Fecha aproximada
    },
    'Telecommunication Services': {
        '2017-04-01': 'Guerra de precios en la industria telecomunicaciones de EE.UU. lleva a caídas en acciones.',  # Fecha aproximada
        '2018-06-12': 'Aprobación de la fusión AT&T-Time Warner, impactando dinámicas del sector.',
        '2020-03-11': 'Pandemia de COVID-19 causa volatilidad inicial pero aumenta demanda de servicios telecom.',
        '2015-08-24': 'Corrección del mercado afecta acciones de telecomunicaciones.',
        '2016-06-24': 'Voto del Brexit genera volatilidad global, incluyendo telecomunicaciones.',
        '2019-01-01': 'Inicio del despliegue de 5G, con costos iniciales presionando acciones.',  # Fecha aproximada
        '2023-01-01': 'Evento hipotético: Nuevas subastas de espectro o cambios regulatorios.'  # Fecha aproximada
    },
    'Utilities': {
        '2021-02-15': 'Fallo de la red eléctrica de Texas durante la tormenta invernal causa volatilidad en acciones.',
        '2022-03-16': 'Inicio de subidas de tasas por la Fed, afectando utilities sensibles a tasas por su alta deuda.',
        '2015-08-24': 'Corrección del mercado impacta acciones de utilities.',
        '2018-12-24': 'Venta masiva afecta todos los sectores, incluyendo utilities.',
        '2020-03-11': 'Pandemia de COVID-19 causa caídas iniciales, aunque utilities son defensivas.',
        '2019-01-29': 'PG&E declara bancarrota debido a incendios en California, afectando acciones de utilities.',
        '2023-01-01': 'Evento hipotético: Nuevas normas de eficiencia energética o mandatos de energías renovables.'  # Fecha aproximada
    }
}

def plot_frequency_critical_dates(extreme_dates, ticker_name, crisis_dates=None):
    """
    Genera un scatter plot de la frecuencia de fechas críticas detectadas, con líneas de eventos de crisis.

    Parameters:
    - extreme_dates (dict): Diccionario de fechas extremas por nivel (de extract_extreme_dates).
    - ticker_name (str): Nombre del ticker para incluir en el título y nombre del archivo.
    - crisis_dates (dict, optional): Diccionario de fechas de crisis {fecha: explicación}.
    """
    # Preparar datos para el scatter plot
    all_dates = []
    date_labels = []
    for coef_name, dates in extreme_dates.items():
        # Solo consideramos los detalles (cD1, cD2, ..., cD4), ignoramos cA
        if 'cD' in coef_name:
            all_dates.extend(dates)
            # Añadimos la etiqueta del detalle (por ejemplo, 'cD1', 'cD2', etc.)
            date_labels.extend([coef_name] * len(dates))
    
    # Contar frecuencias de fechas (considerando todas las bandas de coeficientes)
    date_counts = pd.Series(all_dates).value_counts()
    dates = date_counts.index
    frequencies = date_counts.values
    
    # Crear un diccionario para mapear fechas a sus etiquetas de detalle
    date_to_labels = {}
    for date, label in zip(all_dates, date_labels):
        if date not in date_to_labels:
            date_to_labels[date] = []
        date_to_labels[date].append(label)
    
    # Configurar el gráfico
    plt.figure(figsize=(10, 6))
    
    # Definir colores para cada detalle (cD1 a cD4)
    palette = sns.color_palette('husl', n_colors=4)  # Paleta con 4 colores distintos
    detail_colors = {f'cD{i+1}': palette[i] for i in range(4)}  # cD1, cD2, cD3, cD4
    
    # Scatter plot con colores según el detalle
    for date, freq in zip(dates, frequencies):
        labels = date_to_labels.get(date, [])
        # Si una fecha pertenece a múltiples detalles, usamos el color del detalle más bajo
        if labels:
            lowest_detail = min(labels)  # Por ejemplo, si está en cD1 y cD2, usamos cD1
            color = detail_colors[lowest_detail]
            plt.scatter(date, freq, color=color, s=50, alpha=0.6)
    
    # Líneas verticales para eventos de crisis con texto en la línea
    if crisis_dates:
        for crisis_date, explanation in crisis_dates.items():
            plt.axvline(x=crisis_date, color='blue', linestyle='--', linewidth=1.2)
            # Añadir texto sobre la línea vertical
            plt.text(crisis_date, plt.ylim()[1] * 0.95, explanation, rotation=90, 
                     verticalalignment='top', horizontalalignment='right', 
                     color='black', fontsize=16)
    
    # Configurar etiquetas y título
    plt.title(f'Frequency of Detected Critical Dates - {ticker_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    # El eje y representa el número de veces que una fecha fue identificada como crítica
    # en las diferentes bandas de coeficientes wavelet (cD1, ..., cD4).
    plt.ylabel('Frequency (Number of Detections Across Wavelet Bands)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Crear leyenda para los detalles
    handles = [plt.scatter([], [], color=detail_colors[f'cD{i+1}'], s=50, alpha=0.6, label=f'Detail {i+1}') 
               for i in range(4)]
    plt.legend(handles=handles, loc='upper right', fontsize=10)
    
    # Ajustar límites y diseño
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig(f'plots/frequency_critical_dates_{ticker_name}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Ejemplo de uso (necesitas tener los datos listos)
    ticker_name = 'SP'
    file_path = 'sector_log_returns.csv'
    
    # Cargar datos
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df_log_ret = df['S&P 500'].dropna()
    
    # Ajustar longitud a potencia de 2
    n = len(df_log_ret)
    n_adjusted = 2**int(np.log2(n))
    df_log_ret = df_log_ret.iloc[:n_adjusted]
    
    # Descomposición wavelet
    level = 4
    import pywt
    coeffs = pywt.wavedec(df_log_ret.values, 'haar', level=level, mode='periodization')
    
    # Obtener fechas extremas usando extract_extreme_dates
    top_pct = 3
    extreme_dates = extract_extreme_dates(coeffs, level, df_log_ret.index, top_pct)
    
    # Definir fechas de crisis (formato: fecha -> explicación)
    crisis_dates = {
        pd.to_datetime("2000-01-01"): "2000 Crisis",
        pd.to_datetime("2018-01-01"): "2018 Crisis",
        pd.to_datetime("2020-01-01"): "2020 Crisis"
    }
    
    # Filtrar fechas de crisis que estén dentro del rango de datos
    valid_crisis_dates = {
        date: desc for date, desc in crisis_dates.items()
        if df_log_ret.index.min() <= date <= df_log_ret.index.max()
    }
    
    # Generar el gráfico
    plot_frequency_critical_dates(extreme_dates, ticker_name, valid_crisis_dates)