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

def label_encode_strings_simples(set_datos):
	"""
	Aplica label encoding a la columnas categoricas.
	Devuelve nuevo set de datos.
	"""
	cat_features = ['tipodepropiedad', 'provincia', 'ciudad']
	label_encoder = LabelEncoder()
	nuevo_set_datos = set_datos.copy()
	for cat in cat_features:
	    nuevo_set_datos = nuevo_set_datos.fillna(value = {cat : 'NaN'})
	    nuevo_set_datos[cat] = label_encoder.fit_transform(nuevo_set_datos[cat])
	return nuevo_set_datos

def imputar_nulls_numericos(set_datos):
	"""
	Imputa nulls numericos:
	Reemplaza por el promedio en columnas:
	['metrostotales', 'metroscubiertos', 'antiguedad']
	Reemplaza por cero en columnas:
	['habitaciones', 'banos', 'garages']
	Devuelve nuevo set de datos.
	"""
	cols_con_null_a_cero = ['habitaciones', 'banos', 'garages']
	cols_con_null_a_promedio = ['metrostotales', 'metroscubiertos', 'antiguedad']
	imp_mean = SimpleImputer()
	nuevo_set_datos = set_datos.copy()
	for col in cols_con_null_a_cero:
    		nuevo_set_datos[col] = nuevo_set_datos.fillna(value = {col : 0})  
	for col in cols_con_null_a_promedio:
    		nuevo_set_datos[col] = imp_mean.fit_transform(nuevo_set_datos[[col]])  
	return nuevo_set_datos
	
def busqueda_reportar_mejores_resultados(resultados, n_top=3):
    for i in range(1, n_top + 1):
        candidatos = np.flatnonzero(resultados['rank_test_score'] == i)
        for candidato in candidatos:
            print("Modelo con rango: {0}".format(i))
            print("MEAN: {0:.3f} (STD: {1:.3f})".format(
                  resultados['mean_test_score'][candidato],
                  resultados['std_test_score'][candidato]))
            print("Parametros: {0}".format(resultados['params'][candidato])) # hiper-parametros ?
            print("")
	
def kfold_mostrar_resultados(resultados_kfold):
    print("Resultados: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(resultados_kfold, np.mean(resultados_kfold), np.std(resultados_kfold)))
