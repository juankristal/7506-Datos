#-------------------------------------------------------------
"""
Features para strings, en columnas: 
'titulo, 'descripcion', 'direccion'

Basado en publicacion:
https://www.kaggle.com/lalitparihar44/detailed-text-based-feature-engineering
"""
#-------------------------------------------------------------
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords 
import re
from collections import Counter
import operator

_SOLO_PALABRAS_IMPORTANTES = '_solo_palabras_importantes'
_SIN_SIGNOS_PUNTUACION = '_sin_signos_puntuacion'
_CANTIDAD_PALABRAS_IMPORTANTES = '_cantidad_palabras_importantes'
_CANTIDAD_CARACTERES_IMPORTANTES = '_cantidad_caracteres_en_palabras_importantes'
_LONGITUD_MEDIA_PALABRA  = '_longitud_media_de_palabra'
_CANTIDAD_STOPWORDS = '_cantidad_stopwords'
_CANTIDAD_SIGNOS_PUNTACION = '_cantidad_signos_puntacion'
_CANTIDAD_PALABRAS_MAYUSCULA = '_cantidad_palabras_en_mayuscula'
_CANTIDAD_TITULOS = '_cantidad_titulos'
_CANTIDAD_PALABRAS_TOP_K = '_cantidad_palabras_top_k'
_CANTIDAD_PALABRAS_BOTTOM_K = '_cantidad_palabras_bottom_k'
_CANTIDAD_PREFIJOS_TOP_K  = '_cantidad_prefijos_top_k'
_CANTIDAD_POSTFIJOS_TOP_K = '_cantidad_postfijos_top_k'

TOP_K = 50 # Esperamos muchas repeticiones de la mismas palabras
BOTTOM_K = 10000 # Esperamos muy pocas repeticiones de cada palabra
MIN_LARGO_PALABRA = 3
MAX_LARGO_PREFIJO = 2
MAX_LARGO_POSTFIJO = 2

COLS_STRING = ['titulo', 'descripcion', 'direccion']

#-------------------------------------------------------------

def remove_punctuations_from_string(string1):
    string1 = string1.lower() #changing to lower case
    translation_table = dict.fromkeys(map(ord, string.punctuation), ' ') 
    string2 = string1.translate(translation_table) 
    return string2

def remove_stopwords_from_string(string1):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('spanish')) + r')\b\s*')
    string2 = pattern.sub('', string1)
    return string2
#-------------------------------------------------------------

def fillna_strings_ref(train):    
    train.fillna(value = {
            'titulo' : '', 
            'descripcion' : '', 
            'direccion' : ''
        }, inplace = True) 

def agregar_columnas_auxiliares(train):
    for col in COLS_STRING:
        if not col + _SIN_SIGNOS_PUNTUACION in train.columns:
            train[col + _SIN_SIGNOS_PUNTUACION] =\
                 train[col].apply(
                    lambda x: remove_punctuations_from_string(x))
        if not col + _SOLO_PALABRAS_IMPORTANTES in train.columns:
            train[col + _SOLO_PALABRAS_IMPORTANTES] =\
                train[col + _SIN_SIGNOS_PUNTUACION].apply(
                    lambda x: remove_stopwords_from_string(x))


def agregar_feature_cantidad_palabras_importantes(train):
    for col in COLS_STRING:
        train[col + _CANTIDAD_PALABRAS_IMPORTANTES] = \
            train[col + _SOLO_PALABRAS_IMPORTANTES]\
                .apply(lambda x: len(str(x).split()))


def agregar_feature_cantidad_caracteres_importantes(train):
    for col in COLS_STRING:
        train[col + _CANTIDAD_CARACTERES_IMPORTANTES] =\
            train[col + _SOLO_PALABRAS_IMPORTANTES]\
                .apply(lambda x: len(str(x)))


def agregar_feature_longitud_media_por_palabra(train):
    for col in COLS_STRING:
        train[col + _LONGITUD_MEDIA_PALABRA] =\
            train[col + _CANTIDAD_CARACTERES_IMPORTANTES]\
             /  train[col + _CANTIDAD_PALABRAS_IMPORTANTES]\
                    .transform(lambda x: x if x > 0  else 1) 


def agregar_feature_cantidad_stopwords(train):
    stop_words = set(stopwords.words('spanish'))
    for col in COLS_STRING:
        train[col + _CANTIDAD_STOPWORDS] =\
            train[col + _SIN_SIGNOS_PUNTUACION]\
                .apply(lambda x: len([palabra 
                                        for palabra in str(x).split()
                                            if palabra in stop_words]
                        ))



def agregar_feature_cantidad_signos_puntuacion(train):
    for col in COLS_STRING:
        train[col + _CANTIDAD_SIGNOS_PUNTACION] =\
            train[col]\
                .apply(lambda x: len([caracter 
                                        for caracter in str(x) 
                                            if caracter in string.punctuation]
                        ))



def agregar_feature_cantidad_palabras_mayuscula(train):
    train['direccion' + _CANTIDAD_PALABRAS_MAYUSCULA] =\
        train['direccion']\
            .apply(lambda x: len([palabra 
                                    for palabra in str(x).split() 
                                        if palabra.isupper()]
                    ))


def agregar_feature_cantidad_titulos(train):
    train['direccion' + _CANTIDAD_TITULOS] =\
        train['direccion']\
            .apply(lambda x: len([palabra 
                                    for palabra in str(x).split() 
                                        if palabra.istitle()]
                    ))



def agregar_feature_cantidad_palabras_top_k(train, conteos_palabras_dict):
    top_palabras_dict = {}
    for col in COLS_STRING:
        top_palabras_dict[col] =\
            dict(sorted(
                    conteos_palabras_dict[col].items(), 
                    key=operator.itemgetter(1),reverse=True)[:TOP_K])
    for col in COLS_STRING:
        train[col + _CANTIDAD_PALABRAS_TOP_K] =\
            train[col + _SOLO_PALABRAS_IMPORTANTES].apply(
                lambda x: len([palabra 
                            for palabra in str(x).split() 
                                if palabra in top_palabras_dict[col]]))


def agregar_feature_cantidad_palabras_bottom_k(train, conteos_palabras_dict):
    bottom_palabras_dict = {}
    for col in COLS_STRING:
        bottom_palabras_dict[col] =\
            dict(sorted(
                conteos_palabras_dict[col].items(), 
                key=operator.itemgetter(1),reverse=False)[:BOTTOM_K])
    for col in COLS_STRING:
        train[col + _CANTIDAD_PALABRAS_BOTTOM_K] =\
            train[col + _SOLO_PALABRAS_IMPORTANTES].apply(
                lambda x: len([palabra 
                                for palabra in str(x).split() 
                                    if palabra in bottom_palabras_dict[col]]))

def agregar_feature_cantidad_prefijos_top_k(train, todas_palabras_dict):
    prefijos_palabras_dict = {}
    conteo_prefijos_dict = {}
    for col in COLS_STRING:
        prefijos_palabras_dict[col] =\
            list(map(
                    lambda palabra : palabra[:MAX_LARGO_PREFIJO], 
                    filter(
                        lambda palabra : len(palabra) > MIN_LARGO_PALABRA, 
                        todas_palabras_dict[col].split())
            ))
        conteo_prefijos_dict[col] = Counter(prefijos_palabras_dict[col])
    top_prefijos = {}
    for col in COLS_STRING:
        top_prefijos[col] =\
             dict(sorted(
                    conteo_prefijos_dict[col].items(), 
                    key=operator.itemgetter(1),reverse=True)[:TOP_K])
    for col in COLS_STRING:
        train[col + _CANTIDAD_PREFIJOS_TOP_K] =\
            train[col + _SOLO_PALABRAS_IMPORTANTES].apply(
                lambda x: len([palabra 
                                for palabra in str(x).split() 
                                    if palabra[:MAX_LARGO_PREFIJO] in top_prefijos[col]]))

def agregar_feature_cantidad_postfijos_top_k(train, todas_palabras_dict):
    postfijos_palabras_dict = {}
    conteo_postfijos_dict = {}
    for col in COLS_STRING:
        postfijos_palabras_dict[col] =\
            list(map(
                    lambda palabra : palabra[-MAX_LARGO_POSTFIJO:], 
                    filter(
                        lambda palabra : len(palabra) > MIN_LARGO_PALABRA, 
                        todas_palabras_dict[col].split())
            ))
        conteo_postfijos_dict[col] = Counter(postfijos_palabras_dict[col])
    top_postfijos = {}
    for col in COLS_STRING:
        top_postfijos[col] =\
            dict(sorted(
                conteo_postfijos_dict[col].items(), 
                key=operator.itemgetter(1),reverse=True)[:TOP_K])
    for col in COLS_STRING:
        train[col + _CANTIDAD_POSTFIJOS_TOP_K] =\
            train[col + _SOLO_PALABRAS_IMPORTANTES].apply(
                lambda x: len([palabra 
                                for palabra in str(x).split() 
                                    if palabra[-MAX_LARGO_POSTFIJO:] in top_postfijos[col]]))

def agregar_feature_todos_ref(train):
    fillna_strings_ref(train)
    agregar_columnas_auxiliares(train)
    agregar_feature_cantidad_palabras_importantes(train)
    agregar_feature_cantidad_caracteres_importantes(train)
    agregar_feature_longitud_media_por_palabra(train)
    agregar_feature_cantidad_stopwords(train)
    agregar_feature_cantidad_signos_puntuacion(train)
    agregar_feature_cantidad_palabras_mayuscula(train)
    agregar_feature_cantidad_titulos(train)
    todas_palabras_dict = {}
    conteos_palabras_dict = {}
    for col in COLS_STRING:
        todas_palabras_dict[col] = train[col + _SOLO_PALABRAS_IMPORTANTES].str.cat(sep = ' ')
        conteos_palabras_dict[col] = Counter(re.findall(r"[\w']+", todas_palabras_dict[col]))
    agregar_feature_cantidad_palabras_top_k(train, conteos_palabras_dict)
    agregar_feature_cantidad_palabras_bottom_k(train, conteos_palabras_dict)
    agregar_feature_cantidad_prefijos_top_k(train, todas_palabras_dict)
    agregar_feature_cantidad_postfijos_top_k(train, todas_palabras_dict)

def eliminar_string_no_feature(train):
    for col in COLS_STRING:
        if col in train.columns:
            train.drop([col], axis = 1, inplace = True)
        if col + _SIN_SIGNOS_PUNTUACION in train.columns:
            train.drop([col + _SIN_SIGNOS_PUNTUACION], axis = 1, inplace = True)
        if col + _SOLO_PALABRAS_IMPORTANTES in train.columns:
            train.drop([col + _SOLO_PALABRAS_IMPORTANTES], axis = 1, inplace = True)

def train_agregar_feature_string_todos_df(train):
    df_feature_string = pd.read_csv('data/dima_train_feature_string_0.csv', )
    nuevo_train = train.merge(df_feature_string, on = 'id')
    return nuevo_train

def test_agregar_feature_string_todos_df(test):
    df_feature_string = pd.read_csv('dima_test_feature_string_0.csv', )
    nuevo_test = test.merge(df_feature_string, on = 'id')
    return nuevo_test

def train_string_sin_tags(train):
    

    






























