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