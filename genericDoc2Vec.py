# -*- coding: utf-8 -*-
import numpy as np
import random 
import math
import codecs
from numpy import linalg as LA
try:
    from generic import *
except Exception, e:
    
    def tuplasEntrenamiento(frase, ventana, tags):
        lenFrase = len(frase)
        ventanaAnterior = ventaDelantera = (ventana - 1)/2
        tuplas = []
        for i in range(ventanaAnterior, lenFrase - ventaDelantera, 1):
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

        vector = np.array(vector)
        return vector / LA.norm(vector)

    def randomMatrixGenerator(filas, columnas):
        matriz = []

        for _ in range(filas):
            matriz.append(randomVectorGenerator(columnas))

        return np.array(matriz)

    def softMaxPartial(h, w_jPrima):
        return math.exp(np.dot(h, np.array(w_jPrima).T))

    def doc2vec(frases, npalabras, tagsCount, ventana, alpha, alpha_min, dimensiones, epocas):
        #vectores de las palabras
        palabrasW = []
        paragraphsW = []
        alpha_change = 0.0

        #generacion vectores aleatorios
        for palabra in range(npalabras):
            palabrasW.append(randomVectorGenerator(dimensiones))


        nfrases = len(frases)
        alpha_change = (alpha - alpha_min) / (epocas * nfrases)
        #print alpha_change

        for _ in xrange(tagsCount):
            paragraphsW.append(randomVectorGenerator(dimensiones))

        tuplasTodas = []
        for frase, tags in frases:
            tuplasTodas.append(tuplasEntrenamiento(frase, ventana, tags))


        countVectors = ventana + 1.0
        for _ in xrange(epocas):
            #print _

            for indexFrase, tuplas in enumerate(tuplasTodas):
                if indexFrase % 5000 == 0: print indexFrase
                #generacion de las tuplas de entrenamiento
                #[(palabra a predecir, array de palabras)]
                #tuplas = tuplasEntrenamiento(frase, ventana)
                if tuplas == []:
                    continue

                for palabraIndexTrain, tupla, tags in tuplas:
                    h = np.zeros(dimensiones)
                    for palabra in tupla:
                        h = palabrasW[palabra] + h

                    #introduccion del vector del parrafo
                    for tag in tags:
                        h = h + paragraphsW[tag]
                    #introduccion del contexto del usuario
                    #h = h + context

                    h = h / countVectors #ventana -1 + 1 del vector de la frase + 1 contexto

                    ePowu_j = softMaxPartial(h, palabrasW[palabraIndexTrain])
                    sumsePowu_j = ePowu_j

                    #if not optimiced
                    #for palabraW in palabrasW:
                    #   sumsePowu_j += softMaxPartial(h, palabraW)
                    for nop in xrange(10):
                        index = random.randint(0, npalabras-1)
                        if index != palabraIndexTrain:
                            sumsePowu_j += softMaxPartial(h, palabrasW[index])

                    y_j = ePowu_j/sumsePowu_j
                    E = 0.0 - ePowu_j - math.log(sumsePowu_j)
                    
            
                    palabrasW[palabraIndexTrain] = palabrasW[palabraIndexTrain] - ((alpha * max(0.001, (y_j - 1.0))) * h)
                    palabrasW[palabraIndexTrain] = palabrasW[palabraIndexTrain] / LA.norm(palabrasW[palabraIndexTrain])
                
                    EH_medium = h * ((alpha * E) / (countVectors)) #ventana -1 + 1 del vector de la frase + 1 del contexto
                    #print EH_medium
                    
                    for palabra in tupla:
                        palabrasW[palabra] = palabrasW[palabra] - EH_medium
                        palabrasW[palabra] = palabrasW[palabra] / LA.norm(palabrasW[palabra])

                    for tag in tags:
                        paragraphsW[tag] = paragraphsW[tag] - EH_medium
                        paragraphsW[tag] = paragraphsW[tag] / LA.norm(paragraphsW[tag])
                    
                #actualizacion del alpha
                alpha -= alpha_change
        
        return palabrasW, paragraphsW
    

def readUsersFile(fileName):
    dic = {}
    dicInverse = {}
    dicCount = 0

    #las frases tienen asociadas unas etiquetas, en el valor 1 del array
    frases = []
    dicTags = {}
    dicTagsInverse = {}
    tagsCount = 0
    frasesCount = 0

    with codecs.open(fileName, "r", "utf-8") as fin:
        lastUser = ""
        for linea in fin:
            #exit()
            if len(linea) > 4:
                if "user: " in linea:
                    lastUser = linea.replace("user: ", "").replace("\n", "").replace("\r", "")
                    dicTags[lastUser] = tagsCount
                    dicTagsInverse[str(tagsCount)] = lastUser
                    tagsCount += 1
                else:
                    tweetTag = lastUser + ":" + str(tagsCount) + ":" + str(frasesCount)
                    #print tweetTag
                    dicTags[tweetTag] = tagsCount
                    dicTagsInverse[str(tagsCount)] = tweetTag
                    tagsCount += 1
                    frasesCount += 1

                    frase = ([],(dicTags[lastUser], dicTags[tweetTag]))
                        
                    for palabra in linea.replace("\n", "").replace("\r", "").split(" "):
                        if len(palabra) > 0:
                            if palabra not in dic:
                                dic[palabra] = dicCount
                                dicInverse[str(dicCount)] = palabra
                                dicCount += 1


                            frase[0].append(dic[palabra])

                    frases.append(frase)

    return dic, dicCount, frases, dicInverse, dicTags, dicTagsInverse, tagsCount




if __name__ == '__main__':
    ventana = 5
    alpha = 0.05
    alpha_min = 0.0001
    epocas = 20
    dimensiones = 100

    diccionario, npalabras, frases, diccInverse, dicTags, dicTagsInverse, tagsCount = readUsersFile("exampleUsers")
    
    palabrasW , paragraphsW = doc2vec(frases, npalabras, tagsCount, ventana, alpha, alpha_min, dimensiones, epocas)

    
    calculateCosines = True
    if calculateCosines == True:
        similitudPorUsuario = {}
        for tag in dicTags:
            if ":" not in tag:
                similitudPorUsuario[tag] = {"similitudes": [], "vector": paragraphsW[dicTags[tag]]}

        with open("userVectors.csv", "w") as fout:
            for similitud in similitudPorUsuario:
                for i, elem in enumerate(similitudPorUsuario[similitud]["vector"]):
                    if i == 0:
                        fout.write(str(elem))
                    else:
                        fout.write("," + str(elem))

                fout.write("\n")

        for tag in dicTags:
            if ":" in tag:
                usuario, tagIndex, fraseIndex = tag.split(":")
                vector = paragraphsW[int(tagIndex)]
                vectorUser = similitudPorUsuario[usuario]["vector"]
                #calculamos la similitud
                similitud = np.dot(vectorUser, vector) / (LA.norm(vectorUser) * LA.norm(vector))
                #calculamos la longitud
                fraseIndexes = frases[int(fraseIndex)][0]
                texto = ""
                for index in fraseIndexes:
                    texto += diccInverse[str(index)] + " "

                similitudPorUsuario[usuario]["similitudes"].append((len(texto), similitud))

       
        for usuario in similitudPorUsuario:
            with open("similitudes_"+usuario+".csv", "w") as fout:
                fout.write("longitudFrase;coseno\n")
                for longitud, similitud in similitudPorUsuario[usuario]["similitudes"]:
                    fout.write(str(longitud) + ";" + str(similitud)+ "\n")
    
