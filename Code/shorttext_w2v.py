# -*- coding: utf8 -*-
import csv
from gensim.models.word2vec import Word2Vec
from sklearn.mixture.gmm import GMM, log_multivariate_normal_density
import numpy as np
import preprocess

print_topics = True

# W2V settings
num_dims = 200
num_w2v_iterations = 5

# GMM settings
num_topics = 100
num_gmm_iterations = 100

# train_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/train-PROCESSED-FINAL.csv'
train_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/test-PROCESSED-FINAL.csv'
test_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/test-PROCESSED-FINAL.csv'
output_file_template = "/Volumes/GOFLEX/models/word2vec/{run_id}"

class TweetCSVIterator(object):
	"""Allows for iteration over tweets in a corpus directory."""

	def __init__(self, train_filename):
		# Load in train data
		self.train_filename = train_filename

	def __iter__(self):
		# Load in train data
		with open(self.train_filename, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for row in reader:
				yield row[-1].split()

if __name__ == '__main__':
	w2v_run_id = "w2v_D{dims}_K{topics}_w2vIter{w2v_iter}_gmmIter{gmm_iter}.gensim".format(
		dims=num_dims, topics=num_topics, w2v_iter=num_w2v_iterations, gmm_iter=num_gmm_iterations)
	w2v_output_file = output_file_template.format(run_id=w2v_run_id)

	gmm_run_id = "gmm_D{dims}_K{topics}_w2vIter{w2v_iter}_gmmIter{gmm_iter}.gensim".format(
		dims=num_dims, topics=num_topics, w2v_iter=num_w2v_iterations, gmm_iter=num_gmm_iterations)
	gmm_output_file = output_file_template.format(run_id=gmm_run_id)

	# Create the corpus reader
	print 'Starting W2V training'
	sentences = TweetCSVIterator(train_filename)
	w2v_model = Word2Vec(sentences, min_count=5, size=num_dims, workers=4, iter=num_w2v_iterations) #null_word=True
	# w2v_model.finalize_vocab()
	w2v_model.init_sims(replace=True)
	w2v_model.save(w2v_output_file)

	# Train GMM
	print 'Starting GMM training'
	words = w2v_model.vocab.keys()
	word_vectors = w2v_model.syn0
	gmm_model = GMM(n_components=num_topics, n_iter=num_gmm_iterations, covariance_type='diag')
	gmm_model.fit(word_vectors)

	# Print top topic words
	log_probs = log_multivariate_normal_density(word_vectors, gmm_model.means_, gmm_model.covars_, gmm_model.covariance_type)
	print np.min(log_probs)
	_, num_col = log_probs.shape
	for col in xrange(num_col):
		# Get the likelihood of each word vector under each Gaussian component
		top_n = 10
		log_component_probs = (log_probs[:,col]).T
		sorted_indexes = np.argsort(log_component_probs)[::-1][:top_n]

		# print np.sort(log_component_probs)[:top_n]
		# print np.sort(log_component_probs)[::-1][:top_n]
		# print [log_component_probs[i] for i in sorted_indexes]

		ordered_word_probs = [(w2v_model.index2word[idx], log_component_probs[idx]) for idx in sorted_indexes]

		print '---'
		print "Topic {0}".format(col+1)
		print "Total prob:" + str(sum(log_component_probs))
		print ", ".join(["{w}: {p}".format(w=w, p=p) for w, p in ordered_word_probs])

	print '---'
