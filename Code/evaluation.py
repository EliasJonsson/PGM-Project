import numpy as np
import pandas, pickle, time, bisect
from util import loadData
from itertools import combinations
from scipy.stats import rv_discrete
from operator import itemgetter
from numpy import random
from collections import Counter

def smooth(phiTest, phi):
	"""
	delta-smoothing on self.phiTest
	"""
	minProb = min(min([[w[1] for w in k if w[1] > 0] for k in phiTest]))
	delta = minProb/10.0**4
	NV = len(phiTest)
	MV = len(phiTest[0])
	N = len(phi)
	M = len(phi[0])
	for i in xrange(len(phiTest)):
		for j in xrange(len(phiTest[i])):
			phiTest[i][j][1] = (phiTest[i][j][1]+delta)/(N*M+delta*NV*MV)
	return phiTest

def extractBiterms(X):
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

	type_of_data:	'Training' for the training set, 'Valid' for the validation set.

	"""
	biterms = []
	for tweet in X:
		biterms += list(combinations(tweet, 2))
	B = len(biterms)
	return biterms, B

def perplexity(X, phi, theta, word_to_id, B, K, M, biterms, type_of_data = 'Training'):
	'''
	Returns the perplexity

	Input:
	------------------
	X: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
					   [word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
										.                               
										. 
										.
					   [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]
	type_of_data:   'Training' for the training set, 'Valid' for the validation set.
	'''
	if type_of_data == 'Valid':
		phiOld = phi
		[biterms, B] = extractBiterms(X)
		[phi,word_to_id] = addNewWords(biterms, phi, word_to_id, K, M)
		smooth(phi, phiOld);
	p_B = 0
	for i in xrange(B):
		p_b = 0
		for j in xrange(K):
			wi = biterms[i][0]
			wj = biterms[i][1]
			p_b += theta[j]*phi[j][word_to_id[wi]][1]*phi[j][word_to_id[wj]][1]
		p_B += -(1.0/B)*np.log(p_b)
	return 2**p_B

def addNewWords(biterms, phi, word_to_id, K, M):
	'''
		Add new words into the self.phiTest that appear inside self.bitermsTest
	'''
	phiTest = [[[w[0], w[1]]for w in k] for k in phi]
	k = M;
	word_to_idTest = word_to_id.copy()
	for b in biterms:
		if b[0] not in word_to_idTest:
			word_to_idTest[b[0]] = k
			for i in xrange(K):
				phiTest[i].append([b[0],0])
			k += 1
		if b[1] not in word_to_idTest:
			word_to_idTest[b[1]] = k
			for i in xrange(K):
				phiTest[i].append([b[0],0])
			k += 1
	return phiTest, word_to_idTest;

def inferRun(tweets, bt):
	'''
		Calculate pz_d
	'''
	pz_d = np.zeros((len(tweets),bt.K))
	for i,t in enumerate(tweets):
		pz_d[i] = infer(t, bt)
		if sum(pz_d[i]==0) > 0:
			pz_d[i] += 1*10**-30
			pz_d[i] = pz_d[i]/np.sum(pz_d[i])
	return pz_d

def infer(sent, bt):
	pz_d = np.zeros(bt.K)
	[bs, B]  = extractBiterms([sent])
	cbiterm = Counter(bs)
	sbiterm = sum(cbiterm.values())
	for b in cbiterm:
		w1 = b[0]
		w2 = b[1]
		if (w1 not in bt.word_to_id) or (w2 not in bt.word_to_id):
			continue
		pz_b = np.zeros(bt.K)
		for k in xrange(0,bt.K):
			pz_b[k] = bt.theta[k]*bt.phi[k][bt.word_to_id[w1]][1]*bt.phi[k][bt.word_to_id[w2]][1]
		pz_b = 1.0*pz_b/np.sum(pz_b)
		for k in xrange(0,bt.K):
			pz_d[k] += pz_b[k]#*cbiterm[b]/sbiterm

	return pz_d

def inferRunGMM(tweets, ml,word2index):
	'''
		Calculate pz_d for topic2vec
	'''
	K = len(ml.get_topic_probs())
	pz_d = np.zeros((len(tweets),len(ml.get_topic_probs())))
	for i,t in enumerate(tweets):
		print i
		pz_d[i] = inferGMM(t, ml, word2index)
		if sum(pz_d[i]==0) > 0:
			pz_d[i] += 1*10**-30
			pz_d[i] = pz_d[i]/np.sum(pz_d[i])
	return pz_d

def inferGMM(t, ml, word2index):
	K = len(ml.get_topic_probs())
	pz_d = np.ones(K)
	pw_z = np.exp(ml.get_word_ll_for_topics())
	for k in xrange(K):
		for w in t:
			if w not in word2index:
				continue	
			pz_d[k] *=pw_z[word2index[w]][k]
			#print ml._w2v_model[w][k]
		pz_d[k] *= ml.get_topic_probs()[k]
	pz_d = 1.0*pz_d/np.sum(pz_d)
	return pz_d

		






