from util import loadData
from biterm import Biterm
import numpy as np
import preprocess, time, pickle
#file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'
#file_name = '../Data/training.1600000.processed.noemoticon.csv'
file_name = '../Data/train-PROCESSED-FINAL.csv'
#file_name = '../Data/train-PROCESSED.csv'

tweets = loadData(file_name)

tweets = preprocess.splitWords(tweets)

tweets = tweets

number_of_topics = 100
a = 100.0/number_of_topics
b = 0.001
max_iter = 200
mdl = 16

bt = Biterm(a,b,number_of_topics,max_iter,mdl);
start = time.time()

bt.fit(tweets)
end = time.time()
print end - start
bt.showTopics(10)
[phi, theta] = bt.getParams()
file_name = "model" + str(mdl) + ".pkl"
pickle.dump( bt, open( file_name, "wb" ) )
