import numpy as np
import pandas, pickle, time, bisect
from util import loadData
from itertools import combinations
from scipy.stats import rv_discrete
from operator import itemgetter
from numpy import random


class Biterm(object):
	def __init__(self, a, b, K, max_iter):
		"""
		Initialize the biterm model

		Input:
		-------------
		a:	Dirichlet parameter of the topic distribution, theta ~ Dir(a).
		b:	Dirichlet parameter of the topic-specific word distribtution, phi_z ~ Dir(b).
		K:	Number of topics
		max_iter: Maximum number of iteration for the biterm algorithm

		"""
		self.a = a
		self.b = b
		self.K = K
		self.max_iter = max_iter
		

		self.B = 0
		self.M = 0
		self.biterms = []
		self.topicDist  = None
		self.n_z = np.zeros(K)
		self.word_to_id = dict()
		self.n_wGivenz = None
		self.n_bGivenz = np.zeros(K)


	def fit(self, X):
		"""
		Fit the model according to the given training data.

		Input:
		-------------
		X: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   [word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						   [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]

		Returns
		-------------
		self: object.	
		"""
		N = self.max_iter
		
		self.extractBiterms(X)

		# Initialize topic assignments randomly for all the biterms.
		self.topicDist = np.random.randint(self.K, size = self.B)

		# Initialize n_z, n_wiGivenZ, n_wjGivenZ, the later two are stored in self.n_wGivenz
		self.initCount(X)
		# Train
		for i in xrange(N):
			for j in xrange(len(self.biterms)):
				b = self.biterms[j]
				# Draw a topic for b from P(z|x_b, B, a, b)
				p = np.zeros(self.K)				
				p_num = (self.n_z + self.a)*(self.n_wGivenz[self.word_to_id[b[0]]]+self.b)*(self.n_wGivenz[self.word_to_id[b[1]]]+self.b)
				p_det = self.n_bGivenz+self.M*self.b
				p = p_num/(p_det*p_det)
				topic = self.sample(p)
			
				#Update n_z, n_wiGivenZ, n_wjGivenZ
				self.updateCount(j, topic)
			print i
		self.calcPhi()
		self.calcTheta()
		pickle.dump( [self.n_wGivenz,self.word_to_id, self.n_z], open( "parameters.pkl", "wb" ) )


		return self


	def sample(self, p):
		"""
		Samples from the discrete distrubtion defined with p
		
		Input:
		-------------
		p: list with the discrete probabilities of each topic.
		The probability of topic 1 is first then topic 2 etc.

		Output:
		-------------
		topic: a sample from the distribution
		"""
		r = random.rand()
		p_acc = np.cumsum(p)
		topic =  bisect.bisect_right(p_acc,r*p_acc[-1])
		return topic

	def extractBiterms(self, X):
		"""
		Make a list of all the biterms in the collection.

		Input:
		-------------
		X: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   [word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						   [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]


		"""
		for tweet in X:
			self.biterms += list(combinations(tweet, 2))
		self.B = len(self.biterms)

	def initCount(self, X):
		"""
		initialize the counts, n_z, n_wi|z and n_wj|z.

		Input:
		-------------
		X: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   [word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						   [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]

		"""
		# Initialize n_z
		for i in xrange(self.K):
			self.n_z[i] = sum(self.topicDist == i)

		# Make a set of all unique words.
		word_list = []
		for tweet in X:
			word_list += tweet
		word_set = set(word_list)

		self.M = len(word_set)
		# Initializing word_to_id
		self.word_to_id = dict.fromkeys(word_set, 0)
		i = 0
		for w in self.word_to_id:
			self.word_to_id[w] = i
			i+=1
		# Initialize n_wi|z and n_wj|z. Stored in numpy array.
		self.n_wGivenz = np.zeros((self.M,self.K))

		for i in xrange(self.B):
			topic_id = self.topicDist[i];
			self.n_bGivenz[topic_id] += 1;
			self.n_wGivenz[self.word_to_id[self.biterms[i][0]]][topic_id] += 1;
			self.n_wGivenz[self.word_to_id[self.biterms[i][1]]][topic_id] += 1;


	def updateCount(self, biterm_id, new_topic):
		"""
		Update the counts, n_z, n_wi|z and n_wj|z.

		Input:
		-------------
		biterm_id:	The id of the biterm.
		new_topic:	The biterm's topic.

		"""


		# Update the topic biterm assignment
		old_topic = self.topicDist[biterm_id]
		self.topicDist[biterm_id] = new_topic;
		
		# Update n_z
		self.n_z[old_topic] -= 1
		self.n_z[new_topic] += 1

		# Update n_wi|z
		self.n_wGivenz[self.word_to_id[self.biterms[biterm_id][0]]][old_topic] -= 1
		self.n_wGivenz[self.word_to_id[self.biterms[biterm_id][0]]][new_topic] += 1

		# Update n_wj|z
		self.n_wGivenz[self.word_to_id[self.biterms[biterm_id][1]]][old_topic] -= 1
		self.n_wGivenz[self.word_to_id[self.biterms[biterm_id][1]]][new_topic] += 1

		# Update n_b|z
		self.n_bGivenz[old_topic] -= 1
		self.n_bGivenz[new_topic] += 1


	def calcPhi(self):
		"""
		Calculate phi_w|z. List of K dictionaries.

		"""
		self.phi = [[]]*self.K
		n_wGivenz = self.n_wGivenz.T



		for k in xrange(len(n_wGivenz)):
			sum_n =  sum(n_wGivenz[k])
			for w in self.word_to_id:
				self.phi[k] = self.phi[k] + [[w, 1.0*(n_wGivenz[k][self.word_to_id[w]]+self.b)/(sum_n+self.M*self.b)]]
			self.phi[k] = sorted(self.phi[k], key=itemgetter(1), reverse = True)



	def calcTheta(self):
		"""
		Calculate theta_z.

		"""
		self.theta = np.zeros(self.K)
		for k in xrange(self.K):
			self.theta[k] = (self.n_z[k]+self.a)/(self.B + self.K*self.a)


	def getParams(self):
		"""
		Returns:	The distribution phi_w|z and theta_z

		"""
		return self.phi, self.theta

	def perplexity(self):
		'''
			Returns the perplexity
		'''
		p_B = 0
		for i in xrange(self.B):
			p_b = 0
			for j in xrange(self.K):
				p_b += self.n_z[j]*self.n_wGivenz[self.word_to_id[self.biterms[i][0]]][j]*self.n_wGivenz[self.word_to_id[self.biterms[i][0]]][j]
			p_B += -(1.0/self.B)*np.log(p_b)
		return 2**p_B

	def showTopics(self, n):
		"""
		Shows n words for each topic.

		Input:
		------------------
		n:	number of words to be shown for each topic.

		Output:
		------------------
		Return list of n most descriptive words for each topic.

		"""
		max_phi = [[]]*self.K
		for k in xrange(self.K):
			max_phi[k] = [[row[0], row[1]] for row in self.phi[k][1:n+1]]
		max_phi = [max_phi[i] + [self.theta[i]] for i in xrange(self.K)]
		print max_phi
		df = pandas.DataFrame(max_phi, 
			['Topic ' + str(k+1) for k in xrange(self.K)], 
			['Word ' + str(i+1) if i<n else 'P(topic)' for i in xrange(n+1)]).transpose()
		print df
		file_name = 'data.csv'
		df.to_csv(file_name, sep='\t', encoding='utf-8')












