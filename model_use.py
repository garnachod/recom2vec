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

def getFilmsAndRating():
	users = None
	with open("train/ratings_parsed.txt", "r") as fin:
			users = json.loads(fin.read())

	return users

def isFilmInUser_getRating(user, film):
	for rating in user["ratings"]:
		if rating[0] == film:
			return rating[1]

	return False
if __name__ == '__main__':
	model = Doc2Vec.load("out.model")
	dicMovies = readMoviesData("movies.dat")
	idMovie = "13"
	print "Similar to:" + dicMovies[idMovie].encode("ascii","ignore")
	#objeto de tipo DocvecsArray
	#for film in model.vocab:
	#	print model[film]

	for pelicula, coseno in model.most_similar(positive=["13"], topn=20):
		if math.isnan(coseno) == False:
			print pelicula + "\t\t" + dicMovies[pelicula].encode("ascii","ignore") + "\t\t" + str(coseno)
	

	print "similaridad a la persona 622\n"
	user = getFilmsAndRating()["622"]

	for pelicula, coseno in model.most_similar(positive=model.docvecs[model.docvecs.doctags["u_622"]], topn=20):
		if math.isnan(coseno) == False:
			rating = isFilmInUser_getRating(user, pelicula)
			print "|" + pelicula + "|" + dicMovies[pelicula].encode("ascii","ignore") + "|" + str(coseno) + "|" + str(rating) + "|"