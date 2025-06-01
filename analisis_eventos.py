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
            plt.axvline(x=crisis_date, color='blue', linestyle='--', linewidth=1.2)
            # Añadir texto sobre la línea vertical
            plt.text(crisis_date, plt.ylim()[1] * 0.95, explanation, rotation=90, 
                     verticalalignment='top', horizontalalignment='right', 
                     color='blue', fontsize=8)
    
    # Configurar etiquetas y título
    plt.title(f'Frequency of Detected Critical Dates - {sector_name}', fontsize=14)
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
    plt.savefig(f'plots/frequency_critical_dates_{sector_name}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Definir el diccionario de eventos por sector
    events = {
        'Consumer Discretionary': {
            pd.to_datetime('2012-11-16'): 'Lanzamiento de la Wii U de Nintendo, con ventas iniciales por debajo de las expectativas.',
            pd.to_datetime('2014-10-22'): 'Anuncio de Apple Pay, impulsando el interés en los pagos móviles y el sector tecnológico de consumo.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado debido a preocupaciones sobre la economía china ("Lunes Negro Chino").',
            pd.to_datetime('2015-09-18'): 'Escándalo de emisiones de Volkswagen ("Dieselgate"), afectando gravemente a las acciones automotrices.',
            pd.to_datetime('2017-06-16'): 'Amazon anuncia la adquisición de Whole Foods, sacudiendo el sector minorista y de supermercados.',
            pd.to_datetime('2017-09-18'): 'Toys "R" Us declara bancarrota (Capítulo 11), impactando el sector retail de juguetes.',
            pd.to_datetime('2018-10-15'): 'Sears declara bancarrota, marcando un hito en la crisis del comercio minorista tradicional.',
            pd.to_datetime('2018-12-24'): 'El S&P 500 cierra en su nivel más bajo del año por tensiones comerciales EE.UU.-China y subidas de tasas de la Fed.',
            pd.to_datetime('2020-03-11'): 'OMS declara la pandemia de COVID-19, causando una caída masiva en el gasto discrecional y cierres generalizados.',
            pd.to_datetime('2020-05-04'): 'J.Crew declara bancarrota durante la pandemia, una de las primeras grandes minoristas en hacerlo.',
            pd.to_datetime('2021-03-01'): 'Auge de las "meme stocks" (ej. GameStop, AMC) impulsado por inversores minoristas, generando alta volatilidad.',
            pd.to_datetime('2023-07-21'): 'Estreno de las películas "Barbie" y "Oppenheimer" ("Barbenheimer"), generando un impulso significativo para los cines.'
        },
        'Consumer Staples': {
            pd.to_datetime('2015-03-25'): 'Anuncio de la fusión de Kraft Foods y H.J. Heinz Company.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado ("Lunes Negro Chino") afectando todos los sectores, aunque staples muestra más resiliencia.',
            pd.to_datetime('2016-04-20'): 'Brote de listeria en Blue Bell Creameries lleva a un recall masivo, afectando sus acciones y la confianza del consumidor.',
            pd.to_datetime('2017-06-16'): 'Adquisición de Whole Foods por Amazon intensifica la competencia en el sector de supermercados.',
            pd.to_datetime('2018-01-22'): 'Walmart anuncia aumento de salarios y bonos tras la reforma fiscal en EE.UU., reflejando presiones y reinversión.',
            pd.to_datetime('2018-12-24'): 'Venta masiva en el mercado ("Christmas Eve Crash") impactando precios de acciones, aunque staples relativamente menos afectado.',
            pd.to_datetime('2019-02-22'): 'Acciones de Kraft Heinz caen drásticamente por malos resultados, devaluación de activos e investigación de la SEC.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19: Pánico comprador inicial y acaparamiento de bienes esenciales, luego estabilización de la demanda.',
            pd.to_datetime('2022-02-01'): 'Empresas de bienes de consumo comienzan a reportar fuerte impacto de la inflación en costos y precios al consumidor (tendencia iniciada a finales de 2021).',
            pd.to_datetime('2023-05-01'): 'Continúa la presión inflacionaria y cambios en el comportamiento del consumidor hacia marcas blancas o de menor coste.'
        },
        'Energy': {
            pd.to_datetime('2012-01-01'): 'Auge de la producción de shale oil en EE.UU. comienza a transformar el mercado energético global.',
            pd.to_datetime('2014-11-27'): 'OPEP decide no reducir producción a pesar de la caída de precios, exacerbando el desplome del petróleo.',
            pd.to_datetime('2015-08-24'): 'Precios del petróleo caen significativamente por preocupaciones sobre la desaceleración económica china.',
            pd.to_datetime('2016-01-20'): 'Precios del petróleo alcanzan un mínimo de 12 años (Brent cerca de $27/barril).',
            pd.to_datetime('2019-09-14'): 'Ataques a instalaciones petroleras saudíes (Abqaiq-Khurais) causan la mayor disrupción súbita de suministro de la historia.',
            pd.to_datetime('2020-03-09'): 'Precios del petróleo colapsan tras el desacuerdo OPEP+ y el inicio de la guerra de precios Rusia-Arabia Saudita.',
            pd.to_datetime('2020-04-20'): 'Precios del crudo WTI caen a territorio negativo por primera vez en la historia debido a la falta de almacenamiento.',
            pd.to_datetime('2022-02-24'): 'Invasión de Rusia a Ucrania genera extrema volatilidad en el mercado energético y sanciones contra Rusia.',
            pd.to_datetime('2022-06-08'): 'Precios de la gasolina en EE.UU. alcanzan máximos históricos superando los $5/galón.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Nuevas políticas energéticas impactan el sector.'
        },
        'Financials': {
            pd.to_datetime('2012-05-18'): 'IPO de Facebook enfrenta problemas técnicos y sobrevaloración, afectando la confianza en IPOs tecnológicas y bancos suscriptores.',
            pd.to_datetime('2012-07-26'): 'Mario Draghi (BCE) declara que hará "lo que sea necesario" para preservar el euro, calmando la crisis de deuda soberana europea.',
            pd.to_datetime('2013-05-22'): 'Ben Bernanke (Fed) sugiere posible reducción de compra de bonos ("Taper Tantrum"), causando volatilidad en mercados globales.',
            pd.to_datetime('2013-10-01'): 'Inicio del cierre del gobierno de EE.UU. por 16 días, causando incertidumbre en el mercado.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado ("Lunes Negro Chino") impactando fuertemente a las acciones financieras globales.',
            pd.to_datetime('2016-06-24'): 'Resultados del voto del Brexit generan volatilidad global masiva, especialmente en acciones financieras europeas y británicas.',
            pd.to_datetime('2018-12-24'): 'Fuerte venta masiva en mercados ("Christmas Eve Crash") por temores a subidas de tasas y tensiones comerciales.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 impacta severamente los mercados financieros, con fuertes caídas y aumento de la volatilidad (VIX).',
            pd.to_datetime('2020-03-15'): 'La Reserva Federal recorta las tasas de interés a casi cero y lanza programas de QE masivos.',
            pd.to_datetime('2023-03-10'): 'Colapso de Silicon Valley Bank (SVB) sacude el sector bancario regional en EE.UU. y genera temores de contagio.',
            pd.to_datetime('2023-03-19'): 'UBS acuerda comprar Credit Suisse en un rescate orquestado por las autoridades suizas.'
        },
        'Health Care': {
            pd.to_datetime('2012-06-28'): 'Corte Suprema de EE.UU. ratifica la Ley de Cuidado de Salud Asequible (ACA u "Obamacare"), impactando a aseguradoras y proveedores.',
            pd.to_datetime('2015-09-21'): 'Controversia por el drástico aumento de precios del Daraprim por Turing Pharmaceuticals (Martin Shkreli).',
            pd.to_datetime('2016-08-22'): 'Escándalo por el aumento de precios de EpiPen por Mylan impacta acciones farmacéuticas y genera debate político.',
            pd.to_datetime('2017-07-28'): 'El Senado de EE.UU. vota en contra de derogar partes clave de Obamacare, estabilizando temporalmente el sector.',
            pd.to_datetime('2019-01-01'): 'Aumento de litigios y acuerdos relacionados con la crisis de opioides afectan a grandes farmacéuticas y distribuidoras (la crisis venía desarrollándose años antes).',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19: Caídas iniciales en el sector, seguidas de un fuerte aumento en la demanda de servicios sanitarios, pruebas y desarrollo de vacunas/tratamientos.',
            pd.to_datetime('2020-12-11'): 'FDA otorga la primera autorización de uso de emergencia para una vacuna COVID-19 (Pfizer-BioNTech) en EE.UU.',
            pd.to_datetime('2021-01-01'): 'Desafíos logísticos y de producción en la distribución global de vacunas COVID-19 impactan acciones de salud y la recuperación económica.',
            pd.to_datetime('2022-07-01'): 'La Ley de Reducción de la Inflación (EE.UU.) incluye provisiones para negociar precios de medicamentos, impactando perspectivas de farmacéuticas.',
            pd.to_datetime('2023-06-01'): 'Auge del interés y la inversión en medicamentos para la pérdida de peso (GLP-1), como Ozempic y Wegovy.'
        },
        'Industrials': {
            pd.to_datetime('2015-08-24'): 'Corrección del mercado por preocupaciones económicas globales, especialmente de China, afectando demanda industrial.',
            pd.to_datetime('2016-06-24'): 'Voto del Brexit genera incertidumbre sobre futuras relaciones comerciales e inversiones industriales en Europa y Reino Unido.',
            pd.to_datetime('2018-07-06'): 'EE.UU. impone la primera ronda de aranceles a bienes chinos por valor de $34 mil millones, iniciando una guerra comercial.',
            pd.to_datetime('2019-03-10'): 'Boeing 737 MAX es dejado en tierra a nivel mundial tras dos accidentes fatales, afectando a Boeing y la cadena de suministro aeroespacial.',
            pd.to_datetime('2019-05-10'): 'Escalada de la guerra comercial EE.UU.-China con nuevos aranceles mutuos.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 interrumpe cadenas de suministro globales y paraliza gran parte de la producción industrial.',
            pd.to_datetime('2021-03-23'): 'El buque Ever Given bloquea el Canal de Suez, resaltando la fragilidad de las cadenas de suministro globales.',
            pd.to_datetime('2021-11-15'): 'Firma de la Ley de Inversión en Infraestructura y Empleos en EE.UU., proyectando inversión en transporte, agua, banda ancha.',
            pd.to_datetime('2022-02-24'): 'Conflicto Rusia-Ucrania afecta el comercio global, las cadenas de suministro (especialmente de componentes) y los costos de energía para la industria.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Nuevas regulaciones industriales o políticas económicas.'
        },
        'Information Technology': {
            pd.to_datetime('2012-05-18'): 'Problemática IPO de Facebook genera escepticismo sobre valoraciones en el sector tecnológico.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado ("Lunes Negro Chino") impacta fuertemente al sector tecnológico, especialmente a empresas con exposición a China.',
            pd.to_datetime('2016-06-24'): 'Voto del Brexit genera volatilidad en inversiones tecnológicas y preocupación por el acceso a talento y mercados.',
            pd.to_datetime('2018-03-17'): 'Estalla el escándalo de Cambridge Analytica y Facebook, afectando acciones tecnológicas y aumentando el escrutinio regulatorio.',
            pd.to_datetime('2019-05-13'): 'EE.UU. prohíbe a Huawei el acceso a tecnología estadounidense, afectando cadenas de suministro tecnológicas globales y a empresas de semiconductores.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19: Caídas iniciales en tecnología, seguidas de una fuerte recuperación y auge impulsado por el trabajo remoto, e-commerce y digitalización.',
            pd.to_datetime('2020-10-20'): 'Departamento de Justicia de EE.UU. demanda a Google por prácticas anticompetitivas, marcando un aumento en el escrutinio antimonopolio.',
            pd.to_datetime('2021-04-01'): 'Inicio de la escasez global de semiconductores se vuelve crítica, afectando a múltiples industrias dependientes de la tecnología.',
            pd.to_datetime('2022-01-01'): 'Inicio de una corrección significativa en el sector tecnológico ("Tech Wreck") debido a subidas de tasas de interés y preocupaciones sobre valoraciones altas.',
            pd.to_datetime('2022-11-30'): 'Lanzamiento público de ChatGPT por OpenAI, desatando un frenesí de inversión y desarrollo en inteligencia artificial generativa.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Acciones antimonopolio concertadas contra grandes tecnológicas (Big Tech).'
        },
        'Materials': {
            pd.to_datetime('2015-08-24'): 'Colapso del mercado chino y la devaluación del yuan afectan drásticamente los precios de commodities y el sector de materiales.',
            pd.to_datetime('2016-01-20'): 'Precios de commodities (ej. cobre, mineral de hierro) alcanzan mínimos de varios años por desaceleración económica global y sobreoferta.',
            pd.to_datetime('2018-07-06'): 'Inicio de la guerra comercial EE.UU.-China con imposición de aranceles impacta la demanda y precios de metales industriales y otros materiales.',
            pd.to_datetime('2019-01-25'): 'Colapso de una presa de relaves mineros de Vale en Brumadinho (Brasil), con graves consecuencias humanas y ambientales, impactando al sector minero.',
            pd.to_datetime('2019-05-10'): 'Escalada de tensiones comerciales EE.UU.-China afecta negativamente al sector de materiales por temores a menor demanda global.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 reduce drásticamente la demanda de materiales industriales por paralización de la actividad económica.',
            pd.to_datetime('2021-05-01'): 'Fuerte repunte en los precios de las materias primas (superciclo) debido a la recuperación de la demanda post-pandemia y cuellos de botella.',
            pd.to_datetime('2022-02-24'): 'Conflicto Rusia-Ucrania interrumpe cadenas de suministro de materias primas clave (ej. aluminio, níquel, paladio, fertilizantes) y eleva precios.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Nuevas regulaciones ambientales impactan producción de materiales.',
            pd.to_datetime('2023-08-01'): 'Discusiones sobre la seguridad de las cadenas de suministro de minerales críticos (litio, cobalto, tierras raras) se intensifican globalmente.'
        },
        'Real Estate': {
            pd.to_datetime('2013-05-22'): '"Taper Tantrum": El anuncio de la Fed sobre posible reducción de QE eleva tasas hipotecarias y enfría brevemente el mercado inmobiliario.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado global ("Lunes Negro Chino") afecta temporalmente las inversiones inmobiliarias y REITs.',
            pd.to_datetime('2016-06-24'): 'Voto del Brexit genera incertidumbre en mercados inmobiliarios globales, especialmente en el Reino Unido y Europa.',
            pd.to_datetime('2018-12-19'): 'La Reserva Federal sube las tasas de interés por cuarta vez en el año, afectando las tasas hipotecarias y enfriando el mercado de acciones inmobiliarias.',
            pd.to_datetime('2019-01-01'): 'Desaceleración del mercado de vivienda en varias regiones debido a problemas de asequibilidad y aumento de tasas previo.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 impacta severamente el real estate comercial (oficinas, retail, hoteles) pero impulsa el residencial y logístico.',
            pd.to_datetime('2020-07-01'): 'Éxodo de ciudades hacia suburbios y zonas rurales en algunos países debido al teletrabajo, cambiando la demanda inmobiliaria.',
            pd.to_datetime('2022-03-16'): 'Inicio de un ciclo agresivo de subidas de tasas por la Fed para combatir la inflación, aumentando significativamente los costos de préstamo para bienes raíces.',
            pd.to_datetime('2022-11-01'): 'Caída de los precios de la vivienda en muchos mercados desarrollados después del auge pandémico, debido al alza de tasas.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Cambios en leyes de impuestos o regulaciones de zonificación.',
            pd.to_datetime('2023-05-01'): 'Persistente debilidad en el sector de oficinas comerciales debido al trabajo híbrido, mientras el sector industrial y de datos sigue fuerte.'
        },
        'Telecommunication Services': { # Comunicación a partir de 2018, antes Telecomunicaciones
            pd.to_datetime('2013-07-10'): 'Verizon anuncia la compra de la participación de Vodafone en Verizon Wireless por $130 mil millones, una de las mayores operaciones corporativas.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado ("Lunes Negro Chino") afecta acciones de telecomunicaciones, aunque suelen ser más defensivas.',
            pd.to_datetime('2016-06-24'): 'Voto del Brexit genera volatilidad global, incluyendo telecomunicaciones, por incertidumbre económica.',
            pd.to_datetime('2017-02-01'): 'Intensificación de la guerra de precios en la industria de telecomunicaciones móviles de EE.UU. (ej. T-Mobile Un-carrier) presiona márgenes.',
            pd.to_datetime('2017-12-14'): 'La FCC de EE.UU. vota para derogar las reglas de neutralidad de la red ("Net Neutrality").',
            pd.to_datetime('2018-06-12'): 'Un juez aprueba la fusión AT&T-Time Warner, impactando las dinámicas del sector de medios y telecomunicaciones.',
            pd.to_datetime('2019-01-01'): 'Inicio del despliegue comercial de redes 5G, con altos costos de inversión iniciales presionando a las empresas del sector.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 causa volatilidad inicial pero aumenta drásticamente la demanda de servicios de telecomunicaciones (banda ancha, móviles) por trabajo y ocio en casa.',
            pd.to_datetime('2020-04-01'): 'Fusión de T-Mobile y Sprint es completada en EE.UU., consolidando el mercado móvil.',
            pd.to_datetime('2021-02-24'): 'Subastas de espectro 5G (banda C en EE.UU.) alcanzan precios récord, mostrando el alto valor de esta infraestructura.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Nuevas subastas de espectro o cambios regulatorios significativos.'
        },
        'Utilities': {
            pd.to_datetime('2012-10-29'): 'El huracán Sandy causa apagones masivos en el noreste de EE.UU., destacando la vulnerabilidad de la infraestructura eléctrica.',
            pd.to_datetime('2015-08-24'): 'Corrección del mercado ("Lunes Negro Chino") impacta acciones de utilities, aunque son vistas como sector defensivo.',
            pd.to_datetime('2017-09-20'): 'El huracán María devasta Puerto Rico, destruyendo gran parte de su red eléctrica y generando una crisis humanitaria.',
            pd.to_datetime('2018-12-24'): 'Venta masiva en el mercado ("Christmas Eve Crash") afecta todos los sectores, incluyendo utilities, aunque en menor medida.',
            pd.to_datetime('2019-01-29'): 'PG&E (California) declara bancarrota debido a responsabilidades masivas por incendios forestales causados por sus equipos.',
            pd.to_datetime('2020-03-11'): 'Pandemia de COVID-19 causa caídas iniciales en el mercado, aunque las utilities muestran resiliencia por su naturaleza esencial.',
            pd.to_datetime('2021-02-15'): 'Fallo masivo de la red eléctrica de Texas durante una tormenta invernal extrema causa apagones generalizados y volatilidad en acciones de utilities texanas.',
            pd.to_datetime('2022-03-16'): 'Inicio de subidas de tasas por la Fed para combatir la inflación, afectando a las utilities, sensibles a las tasas por su alta deuda y dividendos.',
            pd.to_datetime('2022-08-16'): 'Firma de la Ley de Reducción de la Inflación en EE.UU., que incluye importantes incentivos para energías limpias, impactando la inversión de las utilities.',
            pd.to_datetime('2023-01-01'): 'Evento hipotético: Nuevas normas de eficiencia energética o mandatos de energías renovables más estrictos.'
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