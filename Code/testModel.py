from util import loadData
from biterm import Biterm
import numpy as np
import evaluation as ev
import preprocess, time, pickle
from w2v_gmm import W2VGMM
from gensim.models.ldamodel import LdaModel

file_name = '../Data/train-PROCESSED-FINAL.csv'

tweets = loadData(file_name)
tweets = preprocess.splitWords(tweets)

beta = [0.001, 0.01, 0.5]
K = [50, 100, 200]
alpha =  [lambda k: 1.0/k, lambda k: 50.0/k, lambda k: 100.0/k]

'''
mdl = 1
for k  in K:
	for a in alpha:
		for b in beta:
			fn = "../BitermData/model" + str(mdl) + ".pkl"
			pkl_file = open(fn, 'rb')
			bt = pickle.load(pkl_file)
			al = a(k)
			print "The perplexity for model " + str(mdl) + "with (beta = " + str(b) + ", alpha = " + str(al) + ", K = " + str(k) + " is " + str(ev.perplexity(tweets, bt.phi, bt.theta, bt.word_to_id, bt.B, bt.K, bt.M, bt.biterms, 'Training'))
			mdl += 1
'''
'''
mdl = 3
fn = "../BitermData/model" + str(mdl) + ".pkl"
pkl_file = open(fn, 'rb')
bt = pickle.load(pkl_file)
pz_d = ev.inferRun(tweets,bt)
print pz_d.shape
file_name = "Pz_d-Biterm3.pkl"
pickle.dump( pz_d, open( file_name, "wb" ) )
 '''


#fn = "Data/gmm_K100_W11_D100.pkl"
#pkl_file = open(fn, 'rb')
model = W2VGMM(100,11 ,200 )
#print model.get_word_ll_for_topics()
word2index = dict()
for i,w in enumerate(model.index2word):
	word2index[w] = i
#print model.get_topic_probs()
pz_d = ev.inferRunGMM(tweets,model,word2index)
file_name = "Pz_d-w2v_gmm.pkl"
pickle.dump( pz_d, open( file_name, "wb" ) )


'''
lda_model = LdaModel.load('../Data/50-K/lda/lda_K50_a10-K_b0.01_iter1000.gensim')
pz_d = np.zeros((len(tweets),50))
for j,tweet in enumerate(tweets):
	BOW = dict(lda_model.get_document_topics(lda_model.id2word.doc2bow(tweet), minimum_probability=0))
	for i in BOW:
		pz_d[j][i] = BOW[i]
	print j
#pz_d = ev.inferRunGMM(tweets,model,word2index)
file_name = "Pz_d-LDA.pkl"
pickle.dump( pz_d, open( file_name, "wb" ) )
'''

'''
lda_model = LdaModel.load('../Data/BEST/lda-u/ldaU_K100_a10-K_b0.01_iter50.gensim')
pz_d = np.zeros((len(tweets),100))
for j,tweet in enumerate(tweets):
	BOW = dict(lda_model.get_document_topics(lda_model.id2word.doc2bow(tweet), minimum_probability=0))
	for i in BOW:
		pz_d[j][i] = BOW[i]
	print j
#pz_d = ev.inferRunGMM(tweets,model,word2index)
file_name = "Pz_d-LDA-U.pkl"
pickle.dump( pz_d, open( file_name, "wb" ) )
'''




