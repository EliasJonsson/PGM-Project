from util import loadData
from biterm import Biterm
import numpy as np
import preprocess
file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'

tweets = loadData(file_name)
tweets = [tweets[i] for i in np.random.permutation(len(tweets))]


tweets = preprocess.removeEllipsis(tweets)
tweets = preprocess.removeClitics(tweets)
tweets = preprocess.removeUsers(tweets)
tweets = preprocess.removePunctuations(tweets)
tweets = preprocess.removeHTMLTags(tweets)
tweets = preprocess.removeURL(tweets)
tweets = preprocess.removePunctuationWords(tweets)
tweets = preprocess.removeShortWords(tweets)
tweets = preprocess.lowercase(tweets)
tweets = preprocess.removeExtraSpaces(tweets)
tweets = preprocess.splitWords(tweets)
tweets = preprocess.removeStopwords(tweets)

a = 0.1
b = 0.1
number_of_topics = 50
max_iter = 100


bt = Biterm(a,b,number_of_topics,max_iter);

bt.fit(tweets)
bt.showTopics(10)
[phi, theta] = bt.getParams()
print theta