from doc2vec import Doc2Vec
from gensim import utils, matutils
from gensim.models.doc2vec import TaggedDocument
from collections import deque
import fileinput
import json
import time
import codecs
import math

def readMoviesData(filename):
	dic = {}

	with codecs.open(filename, "r", "utf-8") as fin:
		for i, line in enumerate(fin):
			if i == 0:
				continue

			lineSplited = line.replace("\n", "").split("\t")
			dic[lineSplited[0]] = lineSplited[1]

	return dic


if __name__ == '__main__':
	model = Doc2Vec.load("out.model")
	dicMovies = readMoviesData("movies.dat")
	idMovie = "1"
	print "Similar to:" + dicMovies[idMovie].encode("ascii","ignore")
	#objeto de tipo DocvecsArray
	for film in model.vocab:
		print model[film]

	for pelicula, coseno in model.most_similar(positive=["1"], topn=20000):
		if math.isnan(coseno) == False:
			print dicMovies[pelicula].encode("ascii","ignore") + "\t\t" + str(coseno)
	

