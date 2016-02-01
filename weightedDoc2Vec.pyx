# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from cpython cimport array
import array
import random 
import math
from numpy import linalg as LA


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef inline DTYPE_t double_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t softMaxPartial_line(np.ndarray a, np.ndarray b): return math.exp(np.dot(a, np.array(b).T))



def tuplasEntrenamiento(frase, ventana, tags):
    lenFrase = len(frase)
    ventanaAnterior = ventaDelantera = (ventana - 1)/2
    tuplas = []
    for i in xrange(ventanaAnterior, lenFrase - ventaDelantera, 1):
        palabrasCercanas = []
        for j in range(i - ventanaAnterior, i + ventanaAnterior + 1):
            if j != i:
                palabrasCercanas.append(frase[j])
        tuplas.append((frase[i], palabrasCercanas, tags))


    return tuplas

def randomVectorGenerator(elementos):
    vector = []
    for _ in range(elementos):
        vector.append((random.random() - 0.5) * 0.2)

    vector = np.array(vector, dtype=DTYPE)
    return vector / LA.norm(vector)

def softMaxPartial(np.ndarray h,np.ndarray w_jPrima,int dimensiones):
    return math.exp(np.dot(h, np.array(w_jPrima).T))


def doc2vec(frases,int npalabras,int tagsCount,int ventana,double alpha,double alpha_min,int dimensiones,int epocas):
    #vectores de las palabras
    palabrasW = []
    paragraphsW = []
    cdef DTYPE_t alpha_change = 0.0

    #generacion vectores aleatorios
    for palabra in range(npalabras):
        palabrasW.append(randomVectorGenerator(dimensiones))

    palabrasW = np.array(palabrasW, dtype=DTYPE)

    nfrases = len(frases)
    alpha_change = (alpha - alpha_min) / (epocas * nfrases)
    #print alpha_change

    for _ in xrange(tagsCount):
        paragraphsW.append(randomVectorGenerator(dimensiones))

    paragraphsW = np.array(paragraphsW, dtype=DTYPE)

    tuplasTodas = [[] for i in xrange(len(frases))]

    for i, elem in enumerate(frases):
        frase, tags = elem
        tuplasTodas[i] = tuplasEntrenamiento(frase, ventana, tags)


    cdef DTYPE_t countVectors = ventana + 1.0
    cdef np.ndarray h = np.zeros(dimensiones, dtype=DTYPE)
    cdef DTYPE_t ePowu_j = 0.0
    cdef DTYPE_t sumsePowu_j = 0.0
    cdef DTYPE_t y_j = 0.0
    cdef DTYPE_t E = 0.0
    cdef DTYPE_t alpha_aux = alpha
    cdef DTYPE_t sumWeights = 0.0

    for _ in xrange(epocas):
        print _

        for indexFrase, tuplas in enumerate(tuplasTodas):
            if indexFrase % 5000 == 0: print indexFrase
            #generacion de las tuplas de entrenamiento
            #[(palabra a predecir, array de palabras)]
            #tuplas = tuplasEntrenamiento(frase, ventana)
            if tuplas == []:
                continue

            for palabraIndexTrain, tupla, tags in tuplas:
                h = np.zeros(dimensiones, dtype=DTYPE)
                
                sumWeights = 0.0
                for palabra, weight in tupla:
                    sumWeights += weight
                    h = (palabrasW[palabra] * weight) + h

                #introduccion del vector del parrafo
                for tag in tags:
                    sumWeights += 1
                    h = h + paragraphsW[tag]
                #introduccion del contexto del usuario
                #h = h + context

                h = h / sumWeights #ventana -1 + 1 del vector de la frase + 1 contexto

                ePowu_j = softMaxPartial_line(h, palabrasW[palabraIndexTrain])
                sumsePowu_j = ePowu_j

                #if not optimiced
                #for palabraW in palabrasW:
                #   sumsePowu_j += softMaxPartial(h, palabraW)
                for index in np.random.randint(0, high=npalabras-1, size=10):
                    #index = random.randint(0, npalabras-1)
                    if index != palabraIndexTrain:
                        sumsePowu_j += softMaxPartial_line(h, palabrasW[index])

                y_j = ePowu_j/sumsePowu_j
                E = 0.0 - ePowu_j - math.log(sumsePowu_j)
                
        
                palabrasW[palabraIndexTrain] = palabrasW[palabraIndexTrain] - ((alpha_aux * double_max(0.001, (y_j - 1.0))) * h)
                palabrasW[palabraIndexTrain] = palabrasW[palabraIndexTrain] / LA.norm(palabrasW[palabraIndexTrain])
            
                aux = ((alpha_aux * E) / sumWeights)
                EH_medium = h * aux #ventana -1 + 1 del vector de la frase + 1 del contexto
                #print EH_medium
                
                for palabra, weight in tupla:
                    palabrasW[palabra] = palabrasW[palabra] - (EH_medium * weight)
                    palabrasW[palabra] = palabrasW[palabra] / LA.norm(palabrasW[palabra])

                for tag in tags:
                    paragraphsW[tag] = paragraphsW[tag] - EH_medium
                    paragraphsW[tag] = paragraphsW[tag] / LA.norm(paragraphsW[tag])
                
            #actualizacion del alpha
            alpha_aux -= alpha_change
    
    return palabrasW, paragraphsW