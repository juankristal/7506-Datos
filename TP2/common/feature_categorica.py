import pandas as pd
import numpy as np

COLS_CATEGORIAS= ['tipodepropiedad', 'provincia', 'ciudad']

def fillna_categorias(train):
	train.fillna(value = {
		'tipodepropiedad' : 'nan', 
		'provincia' : 'nan', 
		'ciudad' : 'nan'}, 
		inplace = True)

def agregar_feature_one_hot_encoding(train):
	fillna_categorias(train)
	nuevo_train = train.copy()
	for col in COLS_CATEGORIAS:
		dummies = pd.get_dummies(train[col])
		dummies = dummies.add_prefix(col + '_')
		nuevo_train = nuevo_train.join(dummies)
	return nuevo_train

def eliminar_categoria_no_feature(train):
	for col in COLS_CATEGORIAS:
		if col in train.columns:
			train.drop([col], axis = 1, inplace = True)

def train_agregar_feature_provincias_ciudades_ohe_reducido_df(train):
	df_feature_ciudades_provincias_ohe_reducido = pd.read_csv('data/dima_train_feature_provincias_ciudades_ohe_reducida.csv')
	nuevo_train = train.merge(df_feature_ciudades_provincias_ohe_reducido, on = 'id')
	return nuevo_train

def test_agregar_feature_provincias_ciudades_ohe_reducido_df(test):
	df_feature_ciudades_provincias_ohe_reducido = pd.read_csv('data/dima_test_feature_provincias_ciudades_ohe_reducida.csv')
	nuevo_test = test.merge(df_feature_ciudades_provincias_ohe_reducido, on = 'id')
	return nuevo_test
