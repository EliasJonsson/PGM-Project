from util import loadData
from biterm import Biterm
import numpy as np
import preprocess, time, pickle, sys
import unicodecsv as csv
#file_name = '../Data/testdata.manualSUBSET.2009.06.14.csv'
# file_name = '../Data/training.1600000.processed.noemoticon.csv'
file_name = '../Data/train-AGGREGATED.csv'
out_file_name = '../Data/train-PROCESSED-FINAL.csv'
tweets = loadData(file_name)
if len(sys.argv) > 1:
	tweets = tweets[0:int(sys.argv[1])]
# tweets = [tweets[i] for i in np.random.permutation(len(tweets))]

print 'Starting preprocessing'
tweets_dict = preprocess.preprocess(tweets)
max_idx = max(tweets_dict.keys())

with open(file_name,'rb') as csvfile:
	reader = csv.reader(csvfile)
	with open(out_file_name, 'wb') as outfile:
		writer = csv.writer(outfile)
		for i, row in enumerate(reader):
			if i > max_idx:
				break
			elif i not in tweets_dict:
				continue

			row[-1] = ' '.join(tweets_dict[i])
			writer.writerow(row)

print 'Done preprocessing'
