from util import loadData
from biterm import Biterm
import numpy as np
import preprocess, time, pickle, sys
#file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'
file_name = '../Data/training.1600000.processed.noemoticon.csv'
n = int(sys.argv[1])
tweets = loadData(file_name)
tweets = tweets[0:n]
tweets = [tweets[i] for i in np.random.permutation(len(tweets))]
tweets = preprocess.preprocess(tweets)
pickle.dump( tweets, open( "preprocessedData.pkl", "wb" ) )