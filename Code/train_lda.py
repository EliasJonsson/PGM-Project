# -*- coding: utf8 -*-
import csv
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
import numpy as np
import preprocess

num_topics_set = [10, 50, 100, 200]
alpha_fracs = [1,10,50,100,250]
betas = [0.001, 0.01, 0.1, 0.5, 1.0]

num_iterations = 100
print_topics = True

train_filename = '../Data/train-PROCESSED-FINAL.csv'
test_filename = '../Data/test-PROCESSED-FINAL.csv'

output_file_template = "/Volumes/GOFLEX/models/lda/{run_id}"


# Load in train data
tweets = []
with open(train_filename, 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in reader:
		tweets += [row[-1]]
texts = [t.split() for t in tweets]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Load in test data
test_tweets = []
with open(test_filename, 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in reader:
		test_tweets += [row[-1]]
test_texts = [t.split() for t in test_tweets]

test_dictionary = corpora.Dictionary(test_texts)
test_corpus = [dictionary.doc2bow(text) for text in test_texts]

# For each configuration...
for num_topics in num_topics_set:
	print "#########"
	print "Topics: "+str(num_topics)
	print "#########"
	for alpha_frac in alpha_fracs:
		print "++++++++"
		print "Alpha: "+str(alpha_frac)+"/K"
		print "++++++++"

		alpha = alpha_frac/float(num_topics)
		for beta in betas:
			print "----------"
			print "Beta: "+str(beta)
			print "----------"

			run_id = "lda_K{K}_a{alpha_frac}-K_b{beta}_iter{iter}.gensim".format(K=num_topics, alpha_frac=alpha_frac, beta=beta, iter=num_iterations)
			print run_id

			output_file = output_file_template.format(run_id=run_id)

			# Train and save
			print 'Training...'
			model = LdaMulticore(corpus, 
				alpha=alpha, eta=beta, passes=50,
				id2word=dictionary, num_topics=num_topics, iterations=num_iterations
			)
			# model.save(output_file)
			print 'Done training'

			# Print top 10 words in topics, if desired
			if print_topics:
				topics = model.show_topics(num_topics=4, formatted=False)
				for topic in topics:
					for tup in topic[1]:
						print tup[0] + ": " + str(tup[1])
					print '\n'

			# Evaluate perplexity
			ll = model.log_perplexity(test_corpus)
			print "LL:   "+str(ll)
			print "Perp: "+str(np.exp2(-ll))
