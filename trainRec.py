# -*- coding: utf-8 -*-
import json
from weigtedDoc2Vec import *

def readFileAnGenerateTrain(filename):
	dicc = {}
	diccInverse = {}
	diccCount = 0
	frases = []
	dicTags = {}
    dicTagsInverse = {}
    tagsCount = 0

	with open(filename, "r") as fin:
		users = json.loads(fin.read())
		#transformacion de los datos, de rating a valor de peso
		for user in users:
			maximum = 0.0
			minimum = 10000.0
			for i in xrange(len(users[user]["ratings"])):
				users[user]["ratings"][i][1] = users[user]["ratings"][i][1] / users[user]["average"]
				if users[user]["ratings"][i][1] > maximum:
					maximum = users[user]["ratings"][i][1]

				if users[user]["ratings"][i][1] < minimum:
					minimum = users[user]["ratings"][i][1]

			
			if maximum != minimum:
				for i in xrange(len(users[user]["ratings"])):
					users[user]["ratings"][i][1] = (users[user]["ratings"][i][1] - minimum) / (maximum - minimum)
			else:
				print str(maximum) + " " + str(minimum)

		#generacion de los diccionarios y demas
		for user in users:
			dicTags[user] = tagsCount
            dicTagsInverse[str(tagsCount)] = user
            tagsCount += 1
			frase = ([],(dicTags[user]))

			for pelicula, peso, tiempo in users[user]["ratings"]:
				if pelicula not in dicc:
					dicc[pelicula] = diccCount
					diccInverse[str(diccCount)] = pelicula
					diccCount += 1

				frase[0].append((dicc[pelicula],peso))
			frases.append(frase)

	return dicc, diccCount, frases, diccInverse, dicTags, dicTagsInverse, tagsCount

if __name__ == '__main__':
	ventana = 15
    alpha = 0.05
    alpha_min = 0.0001
    epocas = 20
    dimensiones = 100

    diccionario, npalabras, frases, diccInverse, dicTags, dicTagsInverse, tagsCount = readFileAnGenerateTrain("exampleUsers")
    palabrasW , paragraphsW = doc2vec(frases, npalabras, tagsCount, ventana, alpha, alpha_min, dimensiones, epocas)