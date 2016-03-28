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
	with open("train/ratings_parsed_test.txt", "r") as fin:
			users = json.loads(fin.read())

	return users

def getFilmsAndRating_train():
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
	idMovie = "2959"
	print "Similar to:" + dicMovies[idMovie].encode("ascii","ignore")
	#objeto de tipo DocvecsArray
	#for film in model.vocab:
	#	print model[film]

	persona = "78"
	user = getFilmsAndRating()[persona]
	user_t = getFilmsAndRating_train()[persona]

	i = 0
	for pelicula, coseno in model.most_similar(positive=[idMovie], topn=100):
		i += 1
		if math.isnan(coseno) == False:
			rating = isFilmInUser_getRating(user, pelicula)
			rating_t = isFilmInUser_getRating(user_t, pelicula)
			#if rating != False or rating_t != False:
			print str(i)+ "|" + pelicula + "|" + dicMovies[pelicula].encode("ascii","ignore") + "|" + str(coseno) + "|" + str(rating) + "|" + str(rating_t) + "|"
	
	

	print "\n\nsimilaridad a la persona %s"%persona
	i = 0
	for pelicula, coseno in model.most_similar(positive=model.docvecs[model.docvecs.doctags["u_"+persona]], topn=100):
		i += 1
		if math.isnan(coseno) == False:
			rating = isFilmInUser_getRating(user, pelicula)
			rating_t = isFilmInUser_getRating(user_t, pelicula)
			if rating != False:
				print str(i)+ "|" + pelicula + "|" + dicMovies[pelicula].encode("ascii","ignore") + "|" + str(coseno) + "|" + str(rating) + "|" + str(rating_t) + "|"

	print "\nsimilaridad a las peliculas que mas han gustado al usuario"
	user_t_sr = sorted(user_t["ratings"], key=lambda x: x[1], reverse=True)

	films = [x[0] for x in user_t_sr[:5]]

	i = 0
	for pelicula, coseno in model.most_similar(positive=films, topn=100):
		i += 1
		if math.isnan(coseno) == False:
			rating = isFilmInUser_getRating(user, pelicula)
			rating_t = isFilmInUser_getRating(user_t, pelicula)
			if rating != False :
				print str(i)+ "|" + pelicula + "|" + dicMovies[pelicula].encode("ascii","ignore") + "|" + str(coseno) + "|" + str(rating) + "|" + str(rating_t) + "|"
