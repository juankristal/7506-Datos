import pandas as pd
import numpy as np

train = pd.read_csv('sets_de_datos/train.csv', index_col = 0)


"""
===========================================================
============ Funciones para agregar features ==============
===========================================================
"""


def agregar_feature_precio_promedio_banos_por_metroscubiertos(df):
    banos = crear_dataframe("banos")
    df = asignar_caracteristica_precio_promedio(df, banos, "banos", "banos_preciopromedio_metroscubiertos")
    return df

def agregar_feature_precio_promedio_habitaciones_por_metroscubiertos(df):
    habitaciones = crear_dataframe("habitaciones")
    df = asignar_caracteristica_precio_promedio(df, habitaciones, "habitaciones", "habitaciones_preciopromedio_metroscubiertos")
    return df

def agregar_feature_precio_promedio_garages_por_metroscubiertos(df):
    garages = crear_dataframe("garages")
    df = asignar_caracteristica_precio_promedio(df, garages, "garages", "garages_preciopromedio_metroscubiertos")
    return df


"""
===========================================================
================== Funciones auxiliares ===================
===========================================================
"""


def crear_dataframe(columna):
    train["intervalo_metroscubiertos"] = train["metroscubiertos"].apply(lambda x: asignar_intervalo(x))
    df = train.groupby(["intervalo_metroscubiertos", columna])["precio"].mean().to_frame()
    df = df.reset_index()
    df = df.pivot_table(values='precio',
                        index=df["intervalo_metroscubiertos"],
                        columns=columna,
                        aggfunc="first")
    train.drop(["intervalo_metroscubiertos"], axis=1, inplace = True)
    return df

def asignar_intervalo(x):
    try:
        return int(x / 50)
    except:
        return np.nan

def asignar_caracteristica_precio_promedio(df, df_caracteristica, caracteristica, nombre):
    df["intervalo_metroscubiertos"] = df["metroscubiertos"].apply(lambda x: asignar_intervalo(x))
    precio_promedio = []
    for index, row in df.iterrows():
        try:
            intervalo = row["intervalo_metroscubiertos"]
            cantidad_caracteristica = row[caracteristica]
            precio = df_caracteristica.loc[intervalo][cantidad_caracteristica]
        except KeyError:
            precio = np.nan
        precio_promedio.append(precio)
    df.drop(["intervalo_metroscubiertos"], axis=1, inplace = True)
    df[nombre] = precio_promedio
    return df