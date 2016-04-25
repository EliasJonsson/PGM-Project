# -*- coding: utf8 -*-
import csv, pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
import numpy as np
import preprocess

# num_topics_set = [50, 100, 200]
# alpha_fracs = [1,10,50,100,250]
# betas = [0.001, 0.01, 0.1, 0.5, 1.0]

num_topics_set = [200]
alpha_fracs = [50]
betas = [0.01]

num_iterations = 50
print_topics = True

train_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/train-PROCESSED-FINAL.csv'
# train_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/test-PROCESSED-FINAL.csv'
test_filename = '/Users/jake/Projects/uoft/csc2515/PGM-Project/Data/test-PROCESSED-FINAL.csv'

output_file_template = "/Volumes/GOFLEX/models/{run_id}"

if __name__ == '__main__':
	pp = pprint.PrettyPrinter(indent=4)

	# Load in train data
	tweets_by_user = {}
	with open(train_filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			if row[3] in tweets_by_user:
				tweets_by_user[row[3]] += [row[-1].split()]
			else:
				tweets_by_user[row[3]] = [row[-1].split()]

	# Aggregate tweets into one document on a per-user basis
	user_docs = []
	for _, user_collection in tweets_by_user.iteritems():
		user_doc = []
		for tweet in user_collection:
			for token in tweet:
				user_doc.append(token)

		user_docs.append(user_doc)
	texts = user_docs

	# pp.pprint(texts)

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	print len(corpus)
	exit()

	# Calculate number of docs per user, on average
	avg_docs = sum([len(user_collection) for user_collection in tweets_by_user])/float(len(tweets_by_user))
	print 'Avg num docs per user: '+str(avg_docs)

	# Calculate average tweet length beforehand
	tweet_lens_before = [len(tweet) for user_collection in tweets_by_user.values() for tweet in user_collection]
	avg_tweet_len_before = sum(tweet_lens_before)/float(len(tweet_lens_before))
	print 'Avg doc len b4: '+str(avg_tweet_len_before)

	# Calculate average tweet length after
	avg_tweet_len_after = sum([len(doc) for doc in user_docs])/float(len(user_docs))
	print 'Avg doc len after: '+str(avg_tweet_len_after)

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

				run_id = "ldaU_K{K}_a{alpha_frac}-K_b{beta}_iter{iter}.gensim".format(K=num_topics, alpha_frac=alpha_frac, beta=beta, iter=num_iterations)
				print run_id

				output_file = output_file_template.format(run_id=run_id)

				# Train and save
				print 'Training...'
				model = LdaMulticore(corpus, 
					alpha=alpha, eta=beta,
					id2word=dictionary, num_topics=num_topics, iterations=num_iterations
				)
				print 'Done training.'
				model.save(output_file)

				# Print top 10 words in topics, if desired
				if print_topics:
					topics = model.show_topics(num_topics=10, formatted=False)
					for topic in topics:
						for tup in topic[1]:
							print tup[0] + ": " + str(tup[1])
						print '\n'

				# Evaluate perplexity
				ll = model.log_perplexity(test_corpus)
				print "LL:   "+str(ll)
				print "Perp: "+str(np.exp2(-ll))
