from util import loadData
from biterm import Biterm
import numpy as np
import preprocess, time, pickle
#file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'
file_name = '../Data/training.1600000.processed.noemoticon.csv'

tweets = loadData(file_name)
tweets = tweets[0:200]
tweets = preprocess.preprocess(tweets)

number_of_topics = 50
a = 1
b = 0.01
max_iter = 300
mdl = 1

bt = Biterm(a,b,number_of_topics,max_iter,mdl);
start = time.time()
bt.fit(tweets)
end = time.time()
print end - start
bt.showTopics(10)
[phi, theta] = bt.getParams()
file_name = "model" + str(mdl) + ".pkl"
pickle.dump( bt, open( file_name, "wb" ) )
