import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
                            'antiguedad': np.float16, \
                            'habitaciones': np.float16, \
                            'garages': np.float16, \
                            'banos': np.float16, \
                            'metroscubiertos': np.float32, \
                            'metrostotales': np.float32, \
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

def eliminar_columnas_complejas(set_datos):
	"""
	PRE: Recibe un set de datos de propiedades en Mexico 
	(train.csv o test.csv)
	POST: Elimina las columnas complejas:
	['titulo', 'descripcion', 'direccion', 'lat', 'lng', 'fecha', 'idzona']
	Devolviendo un nuevo set de datos.
	"""
	drop_cols = ['titulo', 'descripcion', 'direccion', 'lat', 'lng', 'fecha', 'idzona']
	nuevo_set_datos = set_datos.drop(drop_cols, axis=1).copy()
	return nuevo_set_datos

def get_col_nombres_con_raiz(raiz, lista_col_nombres):
    """
    PRE : Recibe un raiz (string), y una lista de 
    cadenas (string).
    POST: Devuelve un nueva lista con las cadenas 
    que llevan la raiz recibido al ppio de la misma 
    """
    lista_col_nombres_con_raiz = []
    for col_nombre in lista_col_nombres:
        col_nombre_split = col_nombre.split('_')
        if col_nombre_split[0] == raiz:
            lista_col_nombres_con_raiz.append(col_nombre)
    return lista_col_nombres_con_raiz

def busqueda_mostrar_resultados_df(cv_result_):
    """
    PRE: Recibe los resultados de un XSearch (cv_results_).
    POST: Muestra los resultados en forma de dataframe.
    """
    lista_col_hiperparam_nombre = get_col_nombres_con_raiz('param', cv_results_)
    df_resultados = pd.DataFrame(cv_results_)
    df_resultados.sort_values(by = 'rank_test_score', inplace = True)
    df_resultados.set_index(['rank_test_score'], inplace = True)
    display(df_resultados[['mean_test_score', 'std_test_score'] + lista_col_hiperparam_nombre])

def busqueda_salvar_resultados_df(cv_result_, nombre_arch):
    """
    PRE: Recibe los resultados de un XSearch (cv_results_).
    POST: Guarda los resultados en forma de csv, para poder cargar 
    como dataframe
    """
    df_resultados = pd.DataFrame(cv_result_)
    df_resultados.to_csv(nombre_arch, index = False)
