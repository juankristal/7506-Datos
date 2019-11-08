#Prueba de un modelo

import pandas as pd
import numpy as np 

#Del ejemplo dado por Navent
def RMSLE(actual, pred):
    return (np.mean((np.log(actual + 1) - np.log(pred + 1)) ** 2)) **.5

#Usado por kaggle
def MAE(valor_real, prediccion):
	return (np.mean(abs(valor_real - prediccion)))

def resultados(valor_real, prediccion, decimales = "{0:.5f}"):
	rmsle = decimales.format(RMSLE(valor_real, prediccion))
	mae = decimales.format(MAE(valor_real, prediccion))

	colA_w = max(len(rmsle), len("RMSLE"))
	colB_w = max(len(mae), len("MAE"))
	total_w = 5 + colA_w + colB_w

	horizontal = "+" + ("-" * total_w) + "+"
	print(horizontal)
	print("| RMSLE" + (" " * (colA_w - 5)) + " | MAE" + (" " * (colB_w - 3)) + " |")
	print(horizontal)
	print("| " + rmsle + (" " * (colA_w - len(rmsle))) + " | " + mae + (" " * (colB_w - len(mae))) + " |")
	print(horizontal)