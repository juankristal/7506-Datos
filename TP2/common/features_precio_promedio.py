import pandas as pd
import numpy as np

train = pd.read_csv('sets_de_datos/train.csv', index_col = 0)


"""
===========================================================
============ Funciones para agregar features ==============
===========================================================
"""
def asignar_precio_promedio_por_cantidad_de_banos_por_ciudad(df):
    df = asignar_precio_promedio_por_cantidad_de_caracteristicas_por_ciudad(df, "banos", "banos_precio_promedio")
    return df


def asignar_precio_promedio_por_cantidad_de_habitaciones_por_ciudad(df):
    df = asignar_precio_promedio_por_cantidad_de_caracteristicas_por_ciudad(df, "habitaciones", "habitaciones_precio_promedio")
    return df


def asignar_precio_promedio_por_cantidad_de_garages_por_ciudad(df):
    df = asignar_precio_promedio_por_cantidad_de_caracteristicas_por_ciudad(df, "garages", "garages_precio_promedio")
    return df


"""
===========================================================
================== Funciones auxiliares ===================
===========================================================
"""


def asignar_precio_promedio_por_cantidad_de_caracteristicas_por_ciudad(df, caracteristica, nombre):
    caracteristica_preciopromedio = train.groupby(["ciudad", caracteristica])["precio"].mean().to_frame()
    caracteristica_preciopromedio = caracteristica_preciopromedio.reset_index()
    caracteristica_preciopromedio = caracteristica_preciopromedio.pivot_table(values="precio",
                                                                              index=caracteristica_preciopromedio["ciudad"],
                                                                              columns=caracteristica,
                                                                              aggfunc="first")
    df = asignar_caracteristica_precio_promedio(df, caracteristica_preciopromedio, caracteristica, nombre)
    return df


def asignar_caracteristica_precio_promedio(df, df_caracteristica, caracteristica, nombre):
    precio_promedio = []
    for index, row in df.iterrows():
        try:
            ciudad = row["ciudad"]
            cantidad_caracteristica = row[caracteristica]
            precio = df_caracteristica.loc[ciudad][cantidad_caracteristica]
        except:
            precio = np.nan
        precio_promedio.append(precio)
    df[nombre] = precio_promedio
    return df