from util import loadData
from biterm import Biterm
import numpy as np
import preprocess, time, pickle
#file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'
file_name = '../Data/training.1600000.processed.noemoticon.csv'

tweets = loadData(file_name)
tweets = tweets[0:400]
#tweets = [tweets[i] for i in np.random.permutation(len(tweets))]
tweets = preprocess.preprocess(tweets)
pickle.dump( tweets, open( "preprocessedData.pkl", "wb" ) )
print tweets



number_of_topics = 50
a = 1
b = 0.01
max_iter = 300


bt = Biterm(a,b,number_of_topics,max_iter);
start = time.time()
bt.fit(tweets)
end = time.time()
print end - start
bt.showTopics(10)
[phi, theta] = bt.getParams()
