import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pywt
from wavelets_analysis import extract_extreme_dates  # Importamos la función necesaria

def load_and_prepare_data(file_path, column_name):
    """
    Carga un archivo CSV y prepara los datos para un sector específico.

    Parameters:
    - file_path (str): Ruta al archivo CSV.
    - column_name (str): Nombre de la columna a usar para log returns.

    Returns:
    - df_log_ret (pd.Series): Serie de log returns.
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df_log_ret = df[column_name].dropna()
    
    # Ajustar longitud a potencia de 2
    n = len(df_log_ret)
    n_adjusted = 2**int(np.log2(n))
    return df_log_ret.iloc[:n_adjusted]

def plot_frequency_critical_dates(extreme_dates, sector_name, crisis_dates=None):
    """
    Genera un scatter plot de la frecuencia de fechas críticas detectadas, con líneas de eventos de crisis.

    Parameters:
    - extreme_dates (dict): Diccionario de fechas extremas por nivel (de extract_extreme_dates).
    - sector_name (str): Nombre del sector para incluir en el título y nombre del archivo.
    - crisis_dates (dict, optional): Diccionario de fechas de crisis {fecha: explicación}.
    """
    # Preparar datos para el scatter plot
    all_dates = []
    date_labels = []
    for coef_name, dates in extreme_dates.items():
        # Solo consideramos los detalles (cD1, cD2, ..., cD4), ignoramos cA
        if 'cD' in coef_name:
            all_dates.extend(dates)
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
        if labels:
            lowest_detail = min(labels)  # Usamos el color del detalle más bajo
            color = detail_colors[lowest_detail]
            plt.scatter(date, freq, color=color, s=50, alpha=0.6)
    
    # Líneas verticales para eventos de crisis con texto en la línea
    if crisis_dates:
        for crisis_date, explanation in crisis_dates.items():
            plt.axvline(x=crisis_date, color='black', linestyle='--', linewidth=1.2)
            # Añadir texto sobre la línea vertical
            plt.text(crisis_date, plt.ylim()[1] * 0.95, explanation, rotation=90, 
                     verticalalignment='top', horizontalalignment='right', 
                     color='black', fontsize=16)
    
    # Configurar etiquetas y título
    plt.title(f'Detected Critical Dates - {sector_name}', fontsize=22)
    plt.xlabel('Date', fontsize=12)
    # El eje y representa el número de veces que una fecha fue identificada como crítica
    # en las diferentes bandas de coeficientes wavelet (cD1, ..., cD4).
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Crear leyenda para los detalles
    handles = [plt.scatter([], [], color=detail_colors[f'cD{i+1}'], s=50, alpha=0.6, label=f'Detail {i+1}') 
               for i in range(4)]
    plt.legend(handles=handles, loc='upper right', fontsize=10)
    
    # Ajustar límites y diseño
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig(f'plots/frequency_critical_dates_{sector_name}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Definir el diccionario de eventos por sector

    events = {
        "Consumer Discretionary": {
            pd.to_datetime("2012-11-16"): "Wii U lanza con ventas débiles.",
            pd.to_datetime("2014-10-22"): "Apple presenta Apple Pay.",
            pd.to_datetime("2015-08-24"): "Mercado cae por temor a China.",
            pd.to_datetime("2015-09-18"): 'Estalla “Dieselgate” de VW.',
            pd.to_datetime("2017-06-16"): "Amazon compra Whole Foods.",
            pd.to_datetime("2017-09-18"): 'Toys "R" Us entra en quiebra.',
            pd.to_datetime("2018-10-15"): "Sears se declara en bancarrota.",
            pd.to_datetime("2018-12-24"): "Sell-off navideño golpea S&P 500.",
            pd.to_datetime("2020-03-11"): "OMS declara pandemia COVID-19.",
            pd.to_datetime("2020-05-04"): "J.Crew quiebra en pandemia.",
            pd.to_datetime("2021-03-01"): "Explota fenómeno meme-stocks.",
            pd.to_datetime("2023-07-21"): "“Barbenheimer” impulsa taquilla."
        },
        "Consumer Staples": {
            pd.to_datetime("2015-03-25"): "Kraft y Heinz anuncian fusión.",
            pd.to_datetime("2015-08-24"): "Caída global; staples resisten.",
            pd.to_datetime("2016-04-20"): "Listeria provoca recall Blue Bell.",
            pd.to_datetime("2017-06-16"): "Amazon-Whole Foods presiona retail.",
            pd.to_datetime("2018-01-22"): "Walmart sube salarios tras reforma.",
            pd.to_datetime("2018-12-24"): "Crash navideño; staples defensivos.",
            pd.to_datetime("2019-02-22"): "Kraft Heinz se desploma por resultados.",
            pd.to_datetime("2020-03-11"): "COVID dispara compras esenciales.",
            pd.to_datetime("2022-02-01"): "Inflación eleva costes de consumo.",
            pd.to_datetime("2023-05-01"): "Consumidores viran a marcas blancas."
        },
        "Energy": {
            pd.to_datetime("2012-01-01"): "Boom shale oil en EE.UU.",
            pd.to_datetime("2014-11-27"): "OPEP no recorta producción.",
            pd.to_datetime("2015-08-24"): "Petróleo baja por temor a China.",
            pd.to_datetime("2016-01-20"): "Brent toca mínimo de 12 años.",
            pd.to_datetime("2019-09-14"): "Ataques paralizan producción saudí.",
            pd.to_datetime("2020-03-09"): "Fracasa pacto OPEP+; precios caen.",
            pd.to_datetime("2020-04-20"): "WTI cotiza en negativo.",
            pd.to_datetime("2022-02-24"): "Invasión rusa sacude energía.",
            pd.to_datetime("2022-06-08"): "Gasolina EE.UU. supera $5/gal.",
            pd.to_datetime("2023-01-01"): "Políticas energéticas hipotéticas."
        },
        "Financials": {
            pd.to_datetime("2012-05-18"): "IPO de Facebook decepciona.",
            pd.to_datetime("2012-07-26"): "Draghi promete salvar el euro.",
            pd.to_datetime("2013-05-22"): "“Taper Tantrum” sacude mercados.",
            pd.to_datetime("2013-10-01"): "Cierre del gobierno en EE.UU.",
            pd.to_datetime("2015-08-24"): "“Lunes Negro” golpea finanzas.",
            pd.to_datetime("2016-06-24"): "Brexit crea volatilidad financiera.",
            pd.to_datetime("2018-12-24"): "Crash navideño hunde bancos.",
            pd.to_datetime("2020-03-11"): "COVID hunde mercados globales.",
            pd.to_datetime("2020-03-15"): "Fed baja tasas a 0 %.",
            pd.to_datetime("2023-03-10"): "Quiebra Silicon Valley Bank.",
            pd.to_datetime("2023-03-19"): "UBS rescata Credit Suisse."
        },
        "Health Care": {
            pd.to_datetime("2012-06-28"): "Corte confirma Obamacare.",
            pd.to_datetime("2015-09-21"): "Shkreli sube precio Daraprim.",
            pd.to_datetime("2016-08-22"): "Polémica por precio de EpiPen.",
            pd.to_datetime("2017-07-28"): "Senado mantiene partes del ACA.",
            pd.to_datetime("2019-01-01"): "Litigios opioides golpean sector.",
            pd.to_datetime("2020-03-11"): "COVID eleva demanda sanitaria.",
            pd.to_datetime("2020-12-11"): "FDA aprueba vacuna Pfizer-BioNTech.",
            pd.to_datetime("2021-01-01"): "Problemas distribución de vacunas.",
            pd.to_datetime("2022-07-01"): "Ley reduce precios de fármacos.",
            pd.to_datetime("2023-06-01"): "Furor por medicamentos GLP-1."
        },
        "Industrials": {
            pd.to_datetime("2015-08-24"): "Caída China reduce demanda.",
            pd.to_datetime("2016-06-24"): "Brexit nubla inversiones.",
            pd.to_datetime("2018-07-06"): "EE.UU. impone aranceles a China.",
            pd.to_datetime("2019-03-10"): "Boeing 737 MAX queda en tierra.",
            pd.to_datetime("2019-05-10"): "Escala guerra comercial.",
            pd.to_datetime("2020-03-11"): "COVID paraliza producción global.",
            pd.to_datetime("2021-03-23"): "Ever Given bloquea Canal Suez.",
            pd.to_datetime("2021-11-15"): "EE.UU. firma Ley de Infraestructura.",
            pd.to_datetime("2022-02-24"): "Guerra Ucrania afecta cadenas.",
            pd.to_datetime("2023-01-01"): "Regulaciones industriales hipotéticas."
        },
        "Information Technology": {
            pd.to_datetime("2012-05-18"): "IPO Facebook sacude tech.",
            pd.to_datetime("2015-08-24"): "“Lunes Negro” golpea tecnología.",
            pd.to_datetime("2016-06-24"): "Brexit preocupa talento tech.",
            pd.to_datetime("2018-03-17"): "Escándalo Cambridge Analytica.",
            pd.to_datetime("2019-05-13"): "EE.UU. veta Huawei.",
            pd.to_datetime("2020-03-11"): "COVID impulsa digitalización.",
            pd.to_datetime("2020-10-20"): "DOJ demanda a Google.",
            pd.to_datetime("2021-04-01"): "Escasez global de chips.",
            pd.to_datetime("2022-01-01"): "‘Tech Wreck’ por alza tasas.",
            pd.to_datetime("2022-11-30"): "Lanzamiento de ChatGPT.",
            pd.to_datetime("2023-01-01"): "Acciones antimonopolio hipotéticas."
        },
        "Materials": {
            pd.to_datetime("2015-08-24"): "Caída China hunde commodities.",
            pd.to_datetime("2016-01-20"): "Metales tocan mínimos plurianuales.",
            pd.to_datetime("2018-07-06"): "Guerra comercial golpea materiales.",
            pd.to_datetime("2019-01-25"): "Tragedia de Brumadinho.",
            pd.to_datetime("2019-05-10"): "Aranceles avivan temores de demanda.",
            pd.to_datetime("2020-03-11"): "COVID baja demanda de materiales.",
            pd.to_datetime("2021-05-01"): "Superciclo eleva precios.",
            pd.to_datetime("2022-02-24"): "Guerra interrumpe suministro.",
            pd.to_datetime("2023-01-01"): "Reglas verdes hipotéticas.",
            pd.to_datetime("2023-08-01"): "Alerta por minerales críticos."
        },
        "Real Estate": {
            pd.to_datetime("2013-05-22"): "“Taper Tantrum” sube hipotecas.",
            pd.to_datetime("2015-08-24"): "Lunes Negro impacta REITs.",
            pd.to_datetime("2016-06-24"): "Brexit crea incertidumbre inmobiliaria.",
            pd.to_datetime("2018-12-19"): "Fed sube tasas; housing se enfría.",
            pd.to_datetime("2019-01-01"): "Desaceleración de vivienda.",
            pd.to_datetime("2020-03-11"): "COVID golpea inmuebles comerciales.",
            pd.to_datetime("2020-07-01"): "Éxodo urbano impulsa suburbios.",
            pd.to_datetime("2022-03-16"): "Fed inicia ciclo de alzas.",
            pd.to_datetime("2022-11-01"): "Precios caen tras auge pandémico.",
            pd.to_datetime("2023-01-01"): "Cambios regulatorios hipotéticos.",
            pd.to_datetime("2023-05-01"): "Oficinas débiles; industrial fuerte."
        },
        "Telecommunication Services": {
            pd.to_datetime("2013-07-10"): "Verizon compra parte de Vodafone.",
            pd.to_datetime("2015-08-24"): "Corrección global; telecom defensivo.",
            pd.to_datetime("2016-06-24"): "Brexit sacude telecom.",
            pd.to_datetime("2017-02-01"): "Guerra de precios móviles.",
            pd.to_datetime("2017-12-14"): "FCC revoca Neutralidad Red.",
            pd.to_datetime("2018-06-12"): "Aprueban fusión AT&T-Time Warner.",
            pd.to_datetime("2019-01-01"): "Arranca despliegue comercial 5G.",
            pd.to_datetime("2020-03-11"): "COVID aumenta demanda de datos.",
            pd.to_datetime("2020-04-01"): "Se completa fusión T-Mobile-Sprint.",
            pd.to_datetime("2021-02-24"): "Subasta 5G banda C récord.",
            pd.to_datetime("2023-01-01"): "Subastas/regulación hipotéticas."
        },
        "Utilities": {
            pd.to_datetime("2012-10-29"): "Huracán Sandy causa apagones.",
            pd.to_datetime("2015-08-24"): "Mercado cae; utilities defensivas.",
            pd.to_datetime("2017-09-20"): "Huracán María daña red PR.",
            pd.to_datetime("2018-12-24"): "Crash navideño golpea utilities.",
            pd.to_datetime("2019-01-29"): "PG&E entra en bancarrota.",
            pd.to_datetime("2020-03-11"): "COVID sacude utilities.",
            pd.to_datetime("2021-02-15"): "Apagón masivo en Texas.",
            pd.to_datetime("2022-03-16"): "Alzas de tasas presionan sector.",
            pd.to_datetime("2022-08-16"): "Ley IRA incentiva energía limpia.",
            pd.to_datetime("2023-01-01"): "Reglas renovables hipotéticas."
        }
    }


# Puedes imprimir una muestra para verificar
# for sector, sector_events in events.items():
# print(f"\n--- {sector} ---")
# for date, event_desc in sorted(sector_events.items()):
# print(f"{date.strftime('%Y-%m-%d')}: {event_desc}")
    
    # Cargar el archivo y obtener la lista de sectores
    file_path = 'sector_log_returns.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    sectors = df.columns.tolist()  # Obtener todas las columnas (sectores)
    
    # Nivel de descomposición wavelet
    level = 4
    top_pct = 3
    
    # Iterar sobre cada sector
    for sector in sectors:
        print(f"Procesando sector: {sector}")
        
        # Cargar y preparar datos para el sector actual
        df_log_ret = load_and_prepare_data(file_path, sector)
        
        # Descomposición wavelet
        coeffs = pywt.wavedec(df_log_ret.values, 'haar', level=level, mode='periodization')
        
        # Obtener fechas extremas usando extract_extreme_dates
        extreme_dates = extract_extreme_dates(coeffs, level, df_log_ret.index, top_pct)
        
        # Obtener eventos para el sector actual (si existen)
        sector_events = events.get(sector, {})
        
        # Filtrar eventos que estén dentro del rango de fechas del sector
        valid_sector_events = {
            date: desc for date, desc in sector_events.items()
            if df_log_ret.index.min() <= date <= df_log_ret.index.max()
        }
        
        # Generar el gráfico para el sector actual
        plot_frequency_critical_dates(extreme_dates, sector, valid_sector_events)
        print(f"Gráfico generado para {sector} y guardado en: plots/frequency_critical_dates_{sector}.png")