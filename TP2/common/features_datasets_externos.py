import pandas as pd
import numpy as np
import datetime as dt

"""
===========================================================
============ Funciones para agregar features ==============
===========================================================
"""


def agregar_feature_incidencia_delictiva(df):
    """Pre: -Columna fecha de tipo datatime
            -No puede haber nan en provincia
    Post: se agrega una columna con cantidad de casos delictivos cada 100000 habitantes.
    El valor depende de la provincia y la fecha de publicación de la casa."""
    incidencia_delictiva = obtener_dataframe_incidencia_delictiva()
    df = asignar_caracteristica_por_fecha(df, incidencia_delictiva, "incidencia_delictiva")
    return df


def agregar_feature_pobreza_porcentual(df):
    """Pre: -Columna fecha de tipo datatime
            -No puede haber nan en provincia
    Post: devuelve el dataframe con la columna la pobreza porcentual agregada"""
    pobreza = obtener_dataframe_pobreza_porcentual()
    df = asignar_caracteristica_por_fecha(df, pobreza, "pobreza_porcentual")
    return df

def agregar_feature_clima(df):
    """Pre: -
    Post: devuelve el datagrame columnas con la temperatura media mínima, temperatura
    media máxima, y si la provincia es de clima seco, templado, cálido y/o frío"""
    clima = obtener_dataframe_clima()
    df = pd.merge(df, clima, on = "provincia", how = "left")
    return df

def agregar_feature_restaurantes(df):
    """Pre: -Columna fecha de tipo datatime
            -No puede haber nan en provincia
    Post: devuelve el dataframe con la columna con la cantidad de restaurantes en la
    provincia y en el año de la casa publicada"""
    restaurantes = obtener_dataframe_restaurantes()
    df = asignar_caracteristica_por_fecha(df, restaurantes, "restaurantes")
    return df

def agregar_feature_pbi(df):
    """Pre: -Columna fecha de tipo datatime
            -No puede haber nan en provincia
    Post: devuelve el dataframe con la columna para cada tipo de pbi (en millones, según
    la provincia y el año de publicación"""
    pbi_2012 = pd.read_csv("data/pbi_2012.csv")
    pbi_2012 = pd.read_csv("data/pbi_2012.csv")
    pbi_2013 = pd.read_csv("data/pbi_2013.csv")
    pbi_2014 = pd.read_csv("data/pbi_2014.csv")
    pbi_2015 = pd.read_csv("data/pbi_2015.csv")
    pbi_2016 = pd.read_csv("data/pbi_2016.csv")
    pbis = [pbi_2012, pbi_2013, pbi_2014, pbi_2015, pbi_2016]
    pbi_campo = crear_dataframe(pbis, "pbi_campo")
    pbi_campo = pbi_campo.set_index("provincia")
    pbi_mineria = crear_dataframe(pbis, "pbi_minería")
    pbi_mineria = pbi_mineria.set_index("provincia")
    pbi_energia_agua_gas = crear_dataframe(pbis, "pbi_energia_agua_gas")
    pbi_energia_agua_gas = pbi_energia_agua_gas.set_index("provincia")
    pbi_construccion = crear_dataframe(pbis, "pbi_construccion")
    pbi_construccion = pbi_construccion.set_index("provincia")
    pbi_industrias_manufactureras = crear_dataframe(pbis, "pbi_industrias_manufactureras")
    pbi_industrias_manufactureras = pbi_industrias_manufactureras.set_index("provincia")
    pbi_comercio = crear_dataframe(pbis, "pbi_comercio")
    pbi_comercio = pbi_comercio.set_index("provincia")
    df = asignar_caracteristica_por_fecha(df, pbi_campo, "pbi_campo")
    df = asignar_caracteristica_por_fecha(df, pbi_mineria, "pbi_mineria")
    df = asignar_caracteristica_por_fecha(df, pbi_energia_agua_gas, "pbi_energia_agua_gas")
    df = asignar_caracteristica_por_fecha(df, pbi_construccion, "pbi_construccion")
    df = asignar_caracteristica_por_fecha(df, pbi_industrias_manufactureras, "pbi_industrias_manufactureras")
    df = asignar_caracteristica_por_fecha(df, pbi_comercio, "pbi_comercio")
    return df

"""
===========================================================
================== Funciones auxiliares ===================
===========================================================
"""


def asignar_caracteristica_por_fecha(train, df, nombre_columna):
    """Recibe el set de datos y un dataset adicional con el index de provincias y columnas con varios años.
    Comprueba para casa en qué año fue publicada y en qué provincia y le asigna el valor de la columna correspondiente a
    esa provincia y año."""
    caracteristicas = []
    train["anio"] = train["fecha"].dt.year
    for index, row in train.iterrows():
        provincia = row["provincia"]
        anio = str(row["anio"])
        caracteristica = df.loc[provincia][anio]
        caracteristicas.append(caracteristica)
    train[nombre_columna] = caracteristicas
    train.drop(["anio"], axis=1, inplace = True)
    return train


def rellenar_año_faltante(df):
    df["2015"].fillna(0)
    columna = []
    for index, row in df.iterrows():
        caracteristica_a = row["2014"]
        caracteristica_b = row["2016"]
        if (pd.isna(caracteristica_a) and  not pd.isna(caracteristica_b)):
            columna.append(caracteristica_b)
        elif (not pd.isna(caracteristica_a) and not pd.isna(caracteristica_b)):
            columna.append(int((int(caracteristica_a) + int(caracteristica_b)) / 2))
        else:
            columna.append(caracteristica_a)
    df["2015"] = columna
    return df


def obtener_dataframe_incidencia_delictiva():
    incidencia_delictiva = pd.read_csv(
        "data/Tasa_de_incidencia_delictiva_por_entidad_federativa_de_ocurrencia_por_cada_cien_mil_habitantes.csv")
    incidencia_delictiva = incidencia_delictiva[["Entidad", "2012 /3", "2013 /4", "2014", "2015", "2016"]]
    incidencia_delictiva.replace({"Baja California": "Baja California Norte",
                                  "Coahuila de Zaragoza": "Coahuila",
                                  "Michoacán de Ocampo": "Michoacán",
                                  "Veracruz de Ignacio de la Llave": "Veracruz",
                                  "México": "Edo. de México",
                                  "Ciudad de México": "Distrito Federal",
                                  "San Luis Potosí": "San luis Potosí"}, inplace=True)
    incidencia_delictiva.columns = ["provincia", "2012", "2013", "2014", "2015", "2016"]
    incidencia_delictiva = incidencia_delictiva.set_index("provincia")
    return incidencia_delictiva


def obtener_dataframe_pobreza_porcentual():
    pobreza = pd.read_csv("data/pobreza_porcentual.csv")
    pobreza["2013"] = (pobreza["2012"] + pobreza["2014"]) / 2
    pobreza["2015"] = (pobreza["2014"] + pobreza["2016"]) / 2
    pobreza.replace({"Baja California": "Baja California Norte",
                     "México": "Edo. de México",
                     "Ciudad de México": "Distrito Federal",
                     "San Luis Potosí": "San luis Potosí"}, inplace=True)
    pobreza = pobreza.set_index("provincia")
    return pobreza

def obtener_dataframe_clima():
    clima = pd.read_csv("data/clima.csv")
    clima.replace({"México": "Edo. de México"}, inplace=True)
    return clima

def obtener_dataframe_restaurantes():
    turismo_2012 = pd.read_csv("data/servicio_turistico_2012.csv")
    turismo_2013 = pd.read_csv("data/servicio_turistico_2013.csv")
    turismo_2014 = pd.read_csv("data/servicio_turistico_2014.csv")
    turismo_2016 = pd.read_csv("data/servicio_turistico_2016.csv")
    turismo_2016.replace({"Distrito Federal\t": "Distrito Federal"}, inplace=True)
    restaurantes = pd.DataFrame(columns=["provincia", "2012", "2013", "2014", "2015", "2016"])
    restaurantes["provincia"] = turismo_2012["provincia"]
    restaurantes["2012"] = turismo_2012["restaurantes"]
    restaurantes["2013"] = turismo_2013["restaurantes"]
    restaurantes["2014"] = turismo_2014["restaurantes"]
    restaurantes["2016"] = turismo_2016["restaurantes"]
    restaurantes.replace({"ND": np.nan}, inplace=True)
    restaurantes = rellenar_año_faltante(restaurantes)
    restaurantes = restaurantes.T.fillna(restaurantes.mean(axis=1)).T
    restaurantes = restaurantes.set_index("provincia")
    return restaurantes

def crear_dataframe(dataframes, columna):
    columnas = ["provincia", "2012", "2013", "2014", "2015", "2016"]
    df = pd.DataFrame(columns=columnas)
    df[columnas[0]] = dataframes[0]["provincia"]
    for i in range (0, len(dataframes)):
        df[columnas[i + 1]] =  dataframes[i][columna]
    return df