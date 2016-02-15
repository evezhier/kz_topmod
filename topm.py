# -*- coding: utf-8 -*-
import csv, sys
from nltk.corpus import stopwords
from string import punctuation
from gensim import corpora, models
from pymystem3 import Mystem

inputfile = sys.argv[1]

sw = stopwords.words("russian")

m = Mystem(entire_input=False)

def preprocessor(line):
	'''
	stop words, pos tags, lemmatization
	'''
	clean = []
	article = line[2]
	tokens = m.lemmatize(article)
	for token in tokens:
		parts = token.split("_")
		if parts[0].strip() not in sw and \
		 token not in punctuation and \
		 not token.isdigit():
			clean.append(token)
	return clean

def create_dictionary(input_file):
	'''
	preprocesses and creates gensim model dictionary
	:param input_file: path to corpus file
	:return: base model dict
	'''
	d = []
	#it is possible to optimize corpus stream for large corpora
	#but it's not essential here
	reader = csv.reader(open(input_file), lineterminator="r")
	#skip first line
	next(reader)
	for line in reader:
		clean = preprocessor(line)
		d.append(clean)
	#print(d)
	dic = corpora.Dictionary(d)
	corpus = [dic.doc2bow(item) for item in d]
	return dic, corpus

def wrap(corpus):
	'''
	creates wrappers to enhance domain precision
	tf-idf performed best
	:param corpus: base model dict
	'''
	
	#wrappers go here
	tfidf = models.TfidfModel(corpus)

	wrapped = tfidf[corpus]
	return wrapped

def lsi_impl(corpus, dictionary):
	'''
	main implementation of LsiModel
	also tried lda but lsi performed better
	:param corpus: corpus or wrapped object
	:param dictionary: base model dict
	'''
	lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
	return lsi.show_topics(num_topics=10, num_words=7,formatted=True)

def writer(filepath, topics):
	with open(filepath, "w", encoding="utf-8") as output:
		for topic in topics:
			output.write("Topic:\n")
			output.write(topic)
			output.write("\n")
	return

def main(filepath):
	'''
	:param filepath: source csv file
	'''
	dic, corpus = create_dictionary(filepath)
	topics = lsi_impl(wrap(corpus), dic)
	print(topics)
	writer(filepath+".topics", topics)
	return ("Done!")

if __name__ == '__main__':
	main(inputfile)
