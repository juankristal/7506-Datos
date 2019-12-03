import pandas as pd
import numpy as np

def agregar_feature_fecha_numerica(train):
    train['fecha_numerica'] =\
        train["fecha"].dt.year * 10000\
         + train["fecha"].dt.month * 100\
          + train["fecha"].dt.day

def eliminar_num_no_feature(train):
    if 'fecha' in train.columns:
        train.drop(['fecha'], axis  = 1, inplace = True)

def completar_lat_lng_con_provincias_y_ciudades(train):
    provincias_dict = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/ciudades_lat_lon.csv')
    ciudades_dict = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/provincias_lat_lon.csv').to_dict()
    train["ciudad"] = train["ciudad"].fillna("")
    train["provincia"] = train["provincia"].fillna("")
    train["lat"] = train.apply(lambda x: x["lat"] if not np.isnan(x["lat"]) else ciudades_dict["Latitude"]\
                           .get(x["ciudad"],provincias_dict["Latitude"].get(x["provincia"], np.nan)), axis=1)
    train["lng"] = train.apply(lambda x: x["lng"] if not np.isnan(x["lng"]) else ciudades_dict["Longitude"]\
                           .get(x["ciudad"],provincias_dict["Longitude"].get(x["provincia"], np.nan)), axis=1)
    return train

def completar_lat_lng_con_idzona_mean(train):
    df_idzona_lat_lng = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/dima_idzona_lat_lng_estadisticas.csv')
    nuevo_train = train.merge(df_idzona_lat_lng, on = 'idzona', how = 'left')
    nuevo_train['lat'] = nuevo_train.apply(
        lambda x: x['lat'] if x['lat'] != np.nan else x['lat_mean'], axis = 1)
    nuevo_train['lng'] = nuevo_train.apply(
        lambda x: x['lng'] if x['lng'] != np.nan else x['lng_mean'], axis = 1)
    nuevo_train = nuevo_train.drop(['lat_mean','lat_median','lat_std',
                                    'lng_mean','lng_median','lng_std'], axis = 1)
    return nuevo_train

def completar_lat_lng_con_idzona_median(train):
    df_idzona_lat_lng = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/dima_idzona_lat_lng_estadisticas.csv')

    nuevo_train = train.merge(df_idzona_lat_lng, on = 'idzona', how = 'left')
    nuevo_train['lat'] = nuevo_train.apply(
        lambda x: x['lat'] if x['lat'] != np.nan else x['lat_median'], axis = 1)
    nuevo_train['lng'] = nuevo_train.apply(
        lambda x: x['lng'] if x['lng'] != np.nan else x['lng_median'], axis = 1)
    nuevo_train = nuevo_train.drop(['lat_mean','lat_median','lat_std',
                                    'lng_mean','lng_median','lng_std'], axis = 1)
    return nuevo_train

def completar_lat_lng_con_promedio_Mexico(train):
    train.fillna(value = {'lat' : 23.062283, 'lng' : -109.699951}, inplace = True)