from sklearn.externals import joblib
from gensim.models.word2vec import Word2Vec
from sklearn.mixture.gmm import log_multivariate_normal_density
import numpy as np

class W2VGMM(object):
	def __init__(self, num_topics, window_size, dim_size, model_folder='../Data/models'):
		models_file_template = model_folder+"/{model}_{run_id}.{filetype}"
		self._run_id = "K{topics}_W{window}_D{dims}".format(topics=num_topics, window=window_size, dims=dim_size)
		w2v_filename = models_file_template.format(model='w2v', run_id=self._run_id, filetype='gensim')
		gmm_filename = models_file_template.format(model='gmm', run_id=self._run_id, filetype='pkl')
		self._w2v_model = Word2Vec.load(w2v_filename)
		self._gmm_model = joblib.load(gmm_filename)

		self.index2word = self._w2v_model.index2word

	def get_word_ll_for_topics(self):
		"""Calculates P(w|z) for all words (rows), given each topic (columns)

		Returns matrix that is VxK (where V is the number of words in the vocabulary, and K is the number of topics)
		"""
		word_vectors = self._w2v_model.syn0
		return log_multivariate_normal_density(word_vectors, self._gmm_model.means_, self._gmm_model.covars_, self._gmm_model.covariance_type)

	def get_topic_probs(self):
		"""Return p(z) for all topics in K (equivalent to the "mixing components")
		"""
		return self._gmm_model.weights_

	def print_topics(self, top_n_words=10):
		log_probs = self.get_word_ll_for_topics()
		_, num_col = log_probs.shape
		for col in xrange(num_col):
			log_component_probs = (log_probs[:,col]).T
			sorted_indexes = np.argsort(log_component_probs)[::-1][:top_n_words]
			ordered_word_probs = [(self._w2v_model.index2word[idx], log_component_probs[idx]) for idx in sorted_indexes]

			print '---'
			print "Topic {0}".format(col+1)
			print "Total prob:" + str(sum(log_component_probs))
			print ", ".join(["{w}: {p}".format(w=w, p=p) for w, p in ordered_word_probs])

		print '---'
