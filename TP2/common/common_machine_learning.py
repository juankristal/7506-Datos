import pandas as pd
import numpy as np

def cargar_set_optimizado(ruta_set_datos, index_col = None):
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
                        date_parser = pd.to_datetime,
			index_col = index_col
                        )
    return df_optimizado
