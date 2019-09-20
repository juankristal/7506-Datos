## Imports

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

 # Configuraciones plots
TAM_TITULO = 35
TAM_ETIQUETA = 30
COLORES_BARRAS = 'colorblind'

## Funciones auxiliares
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

def agregar_serie_boxplot(boxplot, serie, color = 'b', desplazamiento_x = 0, desplazamiento_y = 0):
    """
    PRE: Recibe:
        un boxplot (seaborn.boxplot);
        una serie (pandas.Series) ordenada por
        fila segun se hayan creado las barras
        del boxplot para el dataframe de donde
        proviene la misma;
        un color (string);
        Opcionalmente:
            un desplazamiento en x, y (float)
    POST: Coloca los valores de la serie recibida,
    en el boxplot, haciendolos coincidir con el
    xtick que le corresponde a cada valor.
    Los desplazamientos en x e y sirven para
    terminar de ajustar su posicion.
    Devuelve el boxplot ya configurado.
    """
    posiciones = range(serie.count())
    for pos, xtick in zip(posiciones, boxplot.get_xticks()):
        boxplot.text(
            xtick + desplazamiento_x,
            serie.get_values()[pos] + desplazamiento_y,
            serie.get_values()[pos],
            horizontalalignment = 'center',
            color = color
        )
    return boxplot

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

# Crear dataframes
def crear_dataframe_desde_grupo_porcentajes(df, nombre_columna_grupos, nombre_columna_porcentajes):
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

def crear_dataframe_desde_grupo_estadisticas(df, nombre_columna_grupos, nombre_columna_estadisticas):
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
    EL dataframe devuelto no esta agrupado.
    """
    nuevo_df = df.groupby([nombre_columna_grupos]).agg({nombre_columna_estadisticas : [cuantil_1, 'median', cuantil_3]})
    nuevo_df.columns = nuevo_df.columns.get_level_values(0) + '_' + nuevo_df.columns.get_level_values(1)
    nuevo_df.reset_columns(inplace = True)
    return nuevo_df