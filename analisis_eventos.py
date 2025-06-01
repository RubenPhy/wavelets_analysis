import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    Genera un scatter plot de la frecuencia de fechas críticas detectadas, con líneas de eventos de crisis y umbral.

    Parameters:
    - extreme_dates (dict): Diccionario de fechas extremas por nivel (de extract_extreme_dates).
    - ticker_name (str): Nombre del ticker para el nombre del archivo.
    - crisis_dates (dict, optional): Diccionario de fechas de crisis {nombre: fecha}.
    """
    # Preparar datos para el scatter plot
    all_dates = []
    for coef_name, dates in extreme_dates.items():
        all_dates.extend(dates)
    
    # Contar frecuencias de fechas (considerando todas las bandas de coeficientes)
    date_counts = pd.Series(all_dates).value_counts()
    dates = date_counts.index
    frequencies = date_counts.values
    
    # Configurar el gráfico
    plt.figure(figsize=(10, 6))
    
    # Scatter plot de las fechas críticas
    plt.scatter(dates, frequencies, color='red', s=50, alpha=0.6, label='Critical Dates')
    
    # Línea horizontal del umbral (y=10)
    plt.axhline(y=10, color='black', linestyle='--', linewidth=1.2, label='Threshold y=10')
    
    # Líneas verticales para eventos de crisis
    if crisis_dates:
        for crisis_name, crisis_date in crisis_dates.items():
            plt.axvline(x=crisis_date, color='blue', linestyle='--', linewidth=1.2, 
                        label=f'{crisis_name} Crisis')
    
    # Configurar etiquetas y título
    plt.title('Frequency of detected critical dates', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar leyenda para evitar duplicados
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
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
    
    # Definir fechas de crisis
    crisis_dates = {
        "2000 Crisis": pd.to_datetime("2000-01-01"),
        "2018 Crisis": pd.to_datetime("2018-01-01"),
        "2020 Crisis": pd.to_datetime("2020-01-01")
    }
    
    # Filtrar fechas de crisis que estén dentro del rango de datos
    valid_crisis_dates = {
        name: dt for name, dt in crisis_dates.items()
        if df_log_ret.index.min() <= dt <= df_log_ret.index.max()
    }
    
    # Generar el gráfico
    plot_frequency_critical_dates(extreme_dates, ticker_name, valid_crisis_dates)