from doc2vec import Doc2Vec
from gensim import utils, matutils
from gensim.models.doc2vec import TaggedDocument
from collections import deque
import fileinput
import json
import time


class LabeledLineSentence:
	"""
		ides:
			Number	
			String
	"""
	def __init__(self, source):
		self.source = source
		self.sentences = []
		self.load()

	

	def __iter__(self):
		return self.sentences.__iter__()

	
	def load(self):
		with open(self.source, "r") as fin:
			users = json.loads(fin.read())

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
			palabras_clean = []
			for pelicula, peso, tiempo in users[user]["ratings"]:
				palabras_clean.append((pelicula, peso))

			self.sentences.append(TaggedDocument(palabras_clean, ["u_"+user]))

def train(sentences, save_location, dimension = 128, epochs = 30, method="DBOW"):
		total_start = time.time()
		dm_ = 1 
		if method != "DBOW":
			dm_ = 0
		model = Doc2Vec(min_count=1, window=10, size=dimension, dm = dm_, sample=1e-3, negative=5,workers=8, alpha=0.02)
		
		print "inicio vocab"
		model.build_vocab(sentences)
		print model["1"]
		print "fin vocab"
		first_alpha = model.alpha
		last_alpha = 0.001
		#model.min_alpha = 0.0001
		next_alpha = first_alpha
		for epoch in xrange(epochs):
			start = time.time()
			print "iniciando epoca DBOW:"
			print model.alpha
			next_alpha = (((first_alpha - last_alpha) / float(epochs)) * float(epochs - (epoch+1)) + last_alpha)
			model.min_alpha = next_alpha
			model.train(sentences)
			print model["1"]
			end = time.time()
			model.alpha = next_alpha
			print "tiempo de la epoca " + str(epoch) +": " + str(end - start)

		model.save(save_location)

		total_end = time.time()

		print "tiempo total:" + str((total_end - total_start)/60.0)

if __name__ == '__main__':
	input_path = "train/ratings_parsed.txt"
	sentences = LabeledLineSentence(input_path)
	train(sentences, "out.model")
	

