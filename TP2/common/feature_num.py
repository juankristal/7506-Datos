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