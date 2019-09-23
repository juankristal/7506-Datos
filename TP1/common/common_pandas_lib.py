## Imports

import pandas as pd
import numpy as np
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
                            'tipodepropiedad': 'category', \
                            'provincia': 'category', \
                            'ciudad': 'category', \
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


def setear_titulos_plot(plot, titulo, etiqueta_x, etiqueta_y):
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