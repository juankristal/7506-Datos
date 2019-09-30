## Imports

import pandas as pd
import geopandas
import numpy as np
from math import pi
from matplotlib import pyplot as plt
import seaborn as sns

## Configuraciones plots
TAM_TITULO = 35
TAM_ETIQUETA = 30
COLORES_BARRAS = 'colorblind'
ANCHO_BARRA_LEYENDA = 10

## Funciones auxiliares
 # Carga optimizada del set de datos

def cargar_train_optimizado(ruta_set_datos):
    """
    PRE: Recibe la ruta al el set de datos
    'train.csv', alguno derivado pero que
    mantenga todas sus columnas.
    POST: Carga el set de datos en un dataframe
    (pandas.DataFrame), optimizando los tipos
    de datos, para ahorrar espacio en memoria.
    Devuelve este dataframe (pandas.DataFrame)
    """
    df_optimizado = pd.read_csv(ruta_set_datos, \
                        dtype={ \
                            'id': np.int32, \
                            #'tipodepropiedad': 'category', \
                            #'provincia': 'category', \
                            #'ciudad': 'category', \
                            'antiguedad': np.float16, \
                            'habitaciones': np.float16, \
                            'garages': np.float16, \
                            'banos': np.float16, \
                            'metroscubiertos': np.float16, \
                            'metrostotales': np.float16, \
                            'idzona': np.float32, \
                            'gimnasio': 'bool', \
                            'usosmultiples': 'bool', \
                            'piscina': 'bool', \
                            'escuelascercanas': 'bool', \
                            'centroscomercialescercanos': 'bool', \
                            'precio': np.float32 \
                            },
                        parse_dates = ['fecha'],
                        date_parser = pd.to_datetime
                        )
    return df_optimizado

 # Funciones estadisticas

def cuantil_1(serie):
    """
    PRE: Recibe una serie (pandas.Series) .
    POST: Devuelve el cuantil 3 (75%) de la
    serie recibida .
    """
    return serie.quantile(0.25)

def cuantil_3(serie):
    """
    PRE: Recibe una serie (pandas.Series) .
    POST: Devuelve el cuantil 3 (75%) de la
    serie recibida .
    """
    return serie.quantile(0.75)

def value_counts_normalize_porcentual(serie):
    """
    PRE: Recibe una serie (pandas.Series).
    POST: Devuelve el porcentaje que
    representa cada valor en el conjunto
    total de la serie.
    """
    return serie.value_counts(normalize = True) * 100

 # Funciones para plots

def agregar_serie_plot(plot, serie_valores, serie_y_pos, color = 'k', desplazamiento_x = 0, desplazamiento_y = 0):
    """
    PRE: Recibe:
        un plot (seaborn.plot);
        una serie de valores
        una serie de posiciones en el eje y
        (pandas.Series) ordenada por fila,
        segun como se hayan creado las barras
        del plot para el dataframe de donde
        proviene la misma;
        un color (string);
        Opcionalmente:
            un desplazamiento en x, y (float)
    POST: Coloca los valores de la serie recibida,
    en el plot, haciendolos coincidir con el
    xtick que le corresponde a cada valor, y
    colocandolos en la altura del eje 'y' que
    indica su propio valor.
    Los desplazamientos en x e y sirven para
    terminar de ajustar su posicion.
    Devuelve el plot ya configurado.
    """
    posiciones = range(serie_valores.count())
    for pos, xtick in zip(posiciones, plot.get_xticks()):
        plot.text(
            xtick + desplazamiento_x,
            serie_y_pos.get_values()[pos] + desplazamiento_y,
            serie_valores.round(2).get_values()[pos],
            horizontalalignment = 'center',
            color = color
        )
    return plot

def agregar_valores_stacked_barplot(
        varios_plots_apilados,
        df_pivot,
        color = 'w',
        desplazamiento_x = 0,
        desplazamiento_y = 0
):
    """
    PRE: Recibe una lista de plots apilados que
    representan un "Stacked Barplot" (lista devuelta
    por la funcion plot_stacked_barplot, de este
    modulo); y el dataframe pivoteado de valores
    con el que se creo el stacked barplot.
    EL largo de la lista es igual a la cantidad de
    filas del dataframe recibido.
    POST: Agrega los valores que le corresponden a
    cada barra apilada.
    """
    df_pivot_diff = df_pivot.copy()
    df_pivot_diff = df_pivot_diff.diff()
    df_pivot_diff.iloc[0] = df_pivot.iloc[0]
    for i in range(len(df_pivot.index)):
        plot_i = varios_plots_apilados[i]
        serie_i_valores = df_pivot_diff.iloc[i]
        serie_i_y_pos = df_pivot.iloc[i] / 2
        if (i >= 1):
            serie_i_y_pos = (df_pivot.iloc[i] + df_pivot.iloc[i-1]) / 2
        agregar_serie_plot(plot_i, serie_i_valores, serie_i_y_pos, color, desplazamiento_x, desplazamiento_y)
    return varios_plots_apilados


def setear_titulos_plot(plot, titulo = "", etiqueta_x = "", etiqueta_y = ""):
    """
    PRE: Recibe:
        un plot (seaborn.<algun>plot);
        el titulo del plot (string)
        las etiquetas (string) de los ejes x e y.
    POST: Setea los titulos y etiquetas en el plot,
    con una escale de letra especificada por las
    constantes:
        TAM_TITULO,
        TAM_ETIQUETA
    """
    plot.set_title(titulo, fontsize = TAM_TITULO)
    plot.set_xlabel(etiqueta_x, fontsize = TAM_ETIQUETA)
    plot.set_ylabel(etiqueta_y, fontsize = TAM_ETIQUETA)

 # Funciones para data frames

def agrupar_calcular_porcentajes_desagrupar(df, nombre_columna_grupos, nombre_columna_porcentajes):
    """
    PRE: Recibe un dataframe (pandas.DataFrame), y el nombre de dos
    columnas en el mismo.
    POST: Devuelve un nuevo dataframe, resultante de agrupar (por
    nombre_columna_grupos) el dataframe recibido, y calcular el
    porcentaje que representa cada valor (nombre_columna_porcentajes),
    en cada grupo.
    Las columnas del dataframe resultante seran::
        'nombre_columna_grupos',
        'nombre_columna_porcentajes',
        'porcentaje'
    El data frame devuelto no esta agrupado.
    """
    nuevo_df = df.groupby([nombre_columna_grupos])[nombre_columna_porcentajes].apply(value_counts_normalize_porcentual)
    nuevo_df = nuevo_df.to_frame()
    nuevo_df.columns = ['porcentaje']
    nuevo_df.reset_index(inplace=True)
    return nuevo_df

def agrupar_calcular_estadisticas_desagrupar(df, nombre_columna_grupos, nombre_columna_estadisticas):
    """
    PRE : Recibe un dataframe (pandas.DataFrame) y el nombre de dos
    columnas en el mismo.
    POST: Devuelve un nuevo dataframe, resultante de agrupar (por
    nombre_columna_grupos) el dataframe recibido, y calcular:
        cuantil_1,
        mediana,
        cuantil_3
    (sobre nombre_columna_estadisticas) en cada grupo.
    Las columnas del dataframe resultante seran:
        'nombre_columna_grupo',
        'nombre_columna_estadisticas_cuantil_1',
        'nombre_columna_estadisticas_median',
        'nombre_columna_estadisticas_cuantil_3'
    (en este orden)
    El dataframe devuelto no esta agrupado.
    """
    nuevo_df = df.groupby([nombre_columna_grupos]).agg({nombre_columna_estadisticas : [cuantil_1, 'median', cuantil_3]})
    nuevo_df.columns = nuevo_df.columns.get_level_values(0) + '_' + nuevo_df.columns.get_level_values(1)
    nuevo_df.reset_index(inplace = True)
    return nuevo_df


def columna_bool_a_si_no(df, nombre_columna_bool):
    """
    PRE: Recibe un dataframe (pandas.DataFrame) y el
    nombre de un columna en el mismo, cuyos valores
    sean booleanos.
    POST: Devuelve el mismo dataframe recibido, donde
    los valores del nombre de columna recibido cambian
    de la forma:
        True => 'Si'
        False => 'No'
    """
    df[nombre_columna_bool] = df[nombre_columna_bool].transform(
        lambda x: 'Si' if x == True else 'No'
    )
    return df

 # Funciones para hacer plots
def plot_stacked_barplot(df_pivot, leyenda_titulo = "", leyenda_loc_x = 1, leyenda_loc_y = 1):
    """
    PRE: Recibe un dataframe (pandas.DataFrame),
    pivoteado segun dos columnas cualquiera.
    Ademas puede recibir un titulo para la leyenda,
    y su localizacion.
    POST: Grafica un "Stacked Barplot", donde
    la barras base es la primera linea del dataframe
    recibido, y, el tope, la ultima de ellas.
    El eje x esta definido por el nombre de cada
    columna en el dataframe (y en su mismo orden).
    El eje y esta definido por los valores de la
    "table de pivot".
    Devuelve una lista de plots superpuestos,
    donde el ultimo plot en la misma es el que se
    encuentra ploteado sobre todos los demas.
    """
    fig, ax = plt.subplots()
    varios_plots_apilados = []
    cantidad_filas = len(df_pivot.index)
    for i in range(cantidad_filas):
        varios_plots_apilados.append(
            sns.barplot(
                x = df_pivot.iloc[cantidad_filas - 1 - i].index,
                y = df_pivot.iloc[cantidad_filas - 1 - i].get_values(),
                palette = sns.color_palette(COLORES_BARRAS)[cantidad_filas - 1 - i:cantidad_filas - i],
                ax = ax
            )
        )
    leyenda = plt.legend(df_pivot.index, title = leyenda_titulo, loc = [leyenda_loc_x, leyenda_loc_y])
    for i in range(cantidad_filas):
        color = str(sns.color_palette(COLORES_BARRAS).as_hex()[i])
        leyenda.legendHandles[i].set_color(color)
        leyenda.legendHandles[i].set_lw(ANCHO_BARRA_LEYENDA)
    return varios_plots_apilados


def crear_mapa(series,
               caracteristica,
               vmin,
               vmax,
               titulo,
               titulo_barra,
               color):
    """
    PRE: Recibe :
        una serie indexada por el nombre de las provincias de mexico,
        cuya columna son datos numéricos;
        el nombre de la columna de la serie;
        los valores minimo y maximo para la barra de escalas;
        el titulo del grafico y el titlo de la barra de escalas;
        la paleta de colores para el usar en el grafico.
    POST : Grafica un mapa de Mexico divido por provincias,
    coloreado al dataframe y la columna recibidos.
    Devuelve la figura (matplotlib.pyplot.Figure) y la base
    (matplotlib.pyplot.ax) del grafico. Guarda los gráficos
    en la carpeta Graficos.
    """
    mexico = geopandas.read_file('Data/mexstates.shp') #Los estados pueden ser vistos con mexico.ADMIN_NAME

    #Le pongo los tildes al archivo de estados para que me coincidan con las provincias
    mexico["ADMIN_NAME"].replace({'Nuevo Leon': "Nuevo León",
                               "San Luis Potosi": "San luis Potosí",
                               "Queretaro": "Querétaro",
                               "Yucatan": "Yucatán",
                               "Michoacan": "Michoacán",
                               "Mexico": "Edo. de México",
                               "Baja California": "Baja California Norte"}, inplace=True)

    #Hago un nuevo dataframe con la información del mapa y la antiguedad para cada provincia
    gdf = mexico.set_index("ADMIN_NAME").join(series)
    
    #Grafico el mapa

    #Base donde se va a dibujar
    fig, base = plt.subplots(1, figsize=(10, 6))

    #Si les parece que los ejes están de más, pongan off
    base.axis("on")

    #Pido que me coloreé en base a la caracteristica determinada
    gdf.plot(column=caracteristica, cmap=color, linewidth=0.8, ax=base, edgecolor="0.8")

    #Setteo el título al gráfico
    base.set_title(titulo, fontsize = 23)
    
    #Agrego la barra que indica la antiguedad
    # l:left, b:bottom, w:width, h:height
    cbax = fig.add_axes([1, 0.15, 0.02, 0.65])   
    cbax.set_title(titulo_barra, fontsize = 18)
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbax)
    
    plt.savefig("figs/" + titulo + ".png", bbox_inches = "tight")
    
    my_dpi=65
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)

    return fig, base

def crear_radares_alineados(df, fil, col, paleta_colores):
    """
    PRE: Recibe :
        un dataframe con un index por cada radar que se
        vaya a hacer;
        la cantidad de filas del gráfico;
        la cantidad de columnas del gráfico;
        Ej: si pongo fil=2 y col=3, en una fila voy a
        tener 3 radares y en la siguiente fila 3 radares.
        la paleta de colores con la que se va a colorear
        el gráfico
    POST : Grafica fil * col radares o menos, según los
    datos del datagrame. Devuelve la figura.
    """
    
    # Categorias
    columnas = df.columns
    categorias = list(columnas[1:])
    N = len(categorias)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]
    
    # Color
    paleta = plt.cm.get_cmap(paleta_colores, len(df.index))
    
    lista = []
    
    for fila in range(0, len(df.index)):
        color = paleta(fila)
        
        # Initialise the spider plot
        ax = plt.subplot(fil, col, fila + 1, polar=True, )
    
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
    
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angulos[:-1], categorias, color='grey', size=20)
    
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([1,2,3,4], ["1","2","3","4"], color="grey", size=15)
        plt.ylim(0,4)
        
        # Ind1
        values=df.loc[fila].drop("provincia").values.flatten().tolist()
        values += values[:1]
        ax.plot(angulos, values, color=color, linewidth=1, linestyle='solid')
        ax.fill(angulos, values, color=color, alpha=0.4)
        
        #plt.title(df["provincia"][fila], size=TAM_ETIQUETA, color=color, y=1.1)
        plt.title("{}) {}".format(fila + 1, df["provincia"][fila]), size=TAM_ETIQUETA, color=color, y=1.1)
        
        lista.append(ax)
    
    fig, lista = plt.subplots(0)
    return fig

def crear_radar_superpuestos(categorias, datos_a, datos_b, leyenda_a, leyenda_b, titulo):
    #Basado en https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
    N = len(categorias)
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]
    print(angulos)

    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angulos[:-1], categorias, color='grey')
    
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4], ["1", "2", "3", "4"], color="grey", size=15)
    plt.ylim(0,5)
    
    values = list(datos_a)
    values += values[:1]
    ax.plot(angulos, values, linewidth=2, linestyle='solid', markerfacecolor='blue', label=leyenda_a)
    ax.fill(angulos, values, 'b', alpha=0.1)

    values = list(datos_b)
    values += values[:1]
    ax.plot(angulos, values, linewidth=2, linestyle='solid', markerfacecolor='yellow',label=leyenda_b)
    ax.fill(angulos, values, 'y', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title(titulo, size=TAM_TITULO, y=1.1)
    
def crear_heatmap_porcentaje(df_data, caracteristica, titulo, xlabel, ylabel, color):
    df_data = pd.pivot_table(df_data, index=["provincia"], columns=caracteristica, values="porcentaje")
    df_data = df_data.fillna(0)
    
    hm = sns.heatmap(df_data, linewidths=.5, xticklabels=True, yticklabels=True, cmap = color)
    hm.set_title(titulo, fontsize = TAM_TITULO) 
    hm.set_xlabel(xlabel, fontsize = TAM_ETIQUETA)
    hm.set_ylabel(ylabel, fontsize = TAM_ETIQUETA)
    plt.show()
    
def crear_df_porcentaje_de_provincias(df1, df2, caracteristica):
    """df1: el df de cátedra/
       df2: un df multiindex de la forma {"cantidad": cantidad}, index = [provincia, tipodepropiedad]
       Devuelve un multiindex con el porcentaje para cada tipo de propiedad por provincia"""
    cantidad = df1["provincia"].value_counts()
    index1 = []
    index2 = []
    porcentaje = []
    provincias = set(list(df2.index.get_level_values(0)))
    for provincia in provincias:
        df_provincia = df2.loc[provincia].reset_index()
        for index, row in df_provincia.iterrows():
            index1.append(provincia)
            index2.append(row[caracteristica])
            porcentaje.append((row["cantidad"] * 100) / cantidad[provincia])
    df = pd.DataFrame({"porcentaje": porcentaje}, index = [index1, index2])
    return df

def crear_plot_multiple(data, x, y, caracteristica, paleta, titulo, titulo_barra, xlabel, ylabel, orientacion):
    grafico = sns.barplot(
        x = x, 
        hue = caracteristica,
        y = y,
        data = data,
        palette = paleta,
        orient=orientacion
    )
    grafico.set_title(titulo, fontsize = TAM_TITULO)
    grafico.set_xlabel(xlabel, fontsize = TAM_ETIQUETA)
    grafico.set_ylabel(ylabel, fontsize = TAM_ETIQUETA)
    grafico.legend(title = titulo_barra)
    plt.show()

def densidad_plot_comparativo(df, tipos, columna, caracteristica):
    '''Crea un density plot para ciertos tipos de propiedad en base a una caracteristica. Columna es el tag para la columna de tipos del dataframe'''
    for col in tipos: 
        sns.distplot(df[df[columna] == col][caracteristica].dropna(), hist = False, label = col)
    plt.ylabel('Densidad')
    plt.title(f'Distribucion de {caracteristica} para diferentes tipos de propiedad')

def definir_zona(lat):
    '''Recibe una latitud y devuelve su zona. Lat < 20 para sur, Lat > 24 para norte y entre medio Centro. Devuelve NaN si es NaN'''
    if lat >= 25:
        return 'Norte'
    if lat < 20:
        return 'Sur'
    if 20 < lat < 25:
        return 'Centro'
    return np.nan

def min_threshold(df, threshold):
    '''Funcion para filtrar por una minima cantidad de datos'''
    return df["ciudad"].count() > threshold

def generar_scatter_zonal_oferta(df,zona):
    '''Genera scatters por año para oferta y zona'''
    for año in range(2012,2017):
        nuevo_df = pd.DataFrame()
        info = df.loc[lambda x:(x['año_publicacion'] == año)]\
                                            .groupby(['mes_publicacion','año_publicacion'])
        info = info.agg({'precio':'sum', 'metroscubiertos':'sum', 'mes_publicacion':'count'})
        plt.scatter(info['mes_publicacion'],info['precio']/info['metroscubiertos'], label = año)
        plt.ylabel('Precio')
        plt.xlabel('Cantidad de publicaciones por mes')
        plt.title(f'Relacion oferta/precio por metro cuadrado promedio para zona {zona}')
        plt.legend(loc = 'lower right')
    plt.tight_layout()
    plt.savefig(f'./figs/scatter_oferta_precio_zona_{zona}.jpg')

def barplot_para_no_bools_zonal(df, categoria):
    '''Genera barplots para una categoria de variables discreta en base a un dataframe en base a precios/metrocubierto'''
    aux = df.groupby(['zona', categoria]).agg({'precio':'sum', 'metroscubiertos':'sum'})
    aux['promedio_precio_metro_cubierto'] = aux['precio']/aux['metroscubiertos']
    ax = sns.barplot(x = 'zona', y = 'promedio_precio_metro_cubierto', hue = categoria, order = ['Norte', 'Centro', 'Sur'], data = aux.reset_index())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.title(f'Precio promedio por metro cubierto\n para una cantidad de {categoria} segun zona')
    plt.xlabel('')
    plt.ylabel('Precio')
    plt.savefig(f'./figs/{categoria}-Metrocubierto-Precio barplot.jpg')

def get_frequency(palabras):
    '''En base a una lista de palabras construye un diccionario de frecuencia de cada palabra. Aplica stemming y devuelve un diccionadio de representantes elegidos como la palabra con mas apariciones para una cierta raiz.'''
    spanish_stemmer = SnowballStemmer('spanish')

    distancias = {}
    apariciones = {}

    for palabra in palabras:
        raiz = spanish_stemmer.stem(palabra)
        distancias[raiz] = distancias.get(raiz,{}) #get_closest_word(raiz, distances.get(stemmed_word, palabra), palabra)
        distancias[raiz][palabra] = distancias[raiz].get(palabra, 0) + 1
        apariciones[raiz] = apariciones.get(raiz, 0) + 1
    
    representantes = {}
    for raiz in apariciones:
        representantes[raiz] = max(list(distancias[raiz].items()), key = lambda x:x[1])[0]
              
    return {representantes[raiz]:apariciones[raiz] for raiz in distancias}

def get_palabras_descripciones(df):
    '''Devuelve una lista de palabras de la columna descripcion de un dataframe.'''
    palabras = []
    for texto in list(df['descripcion']):
        texto = str(texto)
        for palabra in texto.split():
            aux = ''
            for letra in palabra:
                if letra.isalpha():
                    aux+=letra
            if aux:
                palabras.append(aux)
    return palabras

def get_stopwords():
    stop_words_sp = set(stopwords.words('spanish'))
    stop_words_en = set(stopwords.words('english'))
    stopwords = stop_words_sp | stop_words_en
    stopwords.add('para')
    spanish_stemmer = SnowballStemmer('spanish')
    return set(map(spanish_stemmer.stem, stopwords))
