import numpy as np
from util import loadQueryData
#from models.ldamodel import LdaModel
import pickle
from collections import Counter
def HScore(pz_d, C):
	intraDis = 0
	K = len(C)
	pk_d = []
	kk = 0
	for k,_ in enumerate(C):
		pk_d.append(pz_d[C[k]])
		for i in xrange(len(pk_d[k])-1):
			for j in xrange(i+1, len(pk_d[k])):
				m = 0.5*(pk_d[k][i]+pk_d[k][j])
				r1 = pk_d[k][i].copy()
				r2 = pk_d[k][j].copy()
				r1[m==0] = 1
				r2[m==0] = 1
				m[m==0] = 1
				log1 = pk_d[k][i]*np.log2(r1/m)
				log2 = pk_d[k][j]*np.log2(r2/m)
				log1[np.isinf(log1)] = 1
				log2[np.isinf(log2)] = 1
				dis = 0.5*np.sum(log1)+0.5*np.sum(log2)
				if np.isnan(dis):
					#print k,i,j,C[k],pk_d[k][i],pk_d[k][j],m
					return
				intraDis += 2*dis/(np.sum(C[k])*(np.sum(C[k])-1))
				kk += 1 
		print k
	print "aaaaaaa"
	print kk,K
	intraDis = (1.0/K)*intraDis
	interDis = 0
	for k1 in xrange(K-1):
		for k2 in xrange(k1+1,K):
			for i in xrange(len(pk_d[k1])):
				m = 0.5(pk_d[k1][i]+pk_d[k2])
				dis = 0.5*np.sum(pk_d[k1][i]*np.log2(pk_d[k1][i]/m),1)+0.5*np.sum(pk_d[k2]*np.log2(pk_d[k2]/m),1)

				interDis += np.sum(dis/(np.sum(C[k2])*np.sum(C[k1])))
		print k1
	print "bbbbbb"
	interDis = 2.0/(K*(K-1))*interDis
	H = 1.0*intraDis/interDis
	return H
fn = "Pz_d-LDA-U.pkl"
#fn = "Pz_d-LDA.pkl"
#fn = "Pz_d-w2v_gmm.pkl"
#fn = "Pz_d-Biterm3.pkl"
pkl_file = open(fn, 'rb')
file_name = '../Data/train-PROCESSED-FINAL.csv'

[tweets, C] = loadQueryData(file_name)
pz_d = pickle.load(pkl_file)
print HScore(pz_d, C)




