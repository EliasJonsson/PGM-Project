# Standard
import re, time, unicodedata
import unicodecsv as csv
from random import shuffle
# Third party
from nltk.twitter import Query, credsfromfile
# Same modules
import preprocess

def strip_emoji(data):
    if not data:
        return data
    if not isinstance(data, basestring):
        return data
    try:
    # UCS-4
        patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return patt.sub(r'', data)

# Define constants
searchfile = '../Data/searches10.txt'
train_outputfile = '../Data/train10.csv'
test_outputfile = '../Data/test10.csv'
author_file = '../Data/authors.csv'

# num_topic_tweets = 1000
num_topic_tweets = 1350
data = []

# fieldnames = ['id_str', 'text', 'retweeted', ('entities', 'hashtags'), ('user', 'screen_name')]
author_names = set()
tweet_ids = set()

# {
# 	u'contributors': None,
# 	u'truncated': False,
# 	u'text': u'Fuck coachella rn',
# 	u'is_quote_status': False,
# 	u'in_reply_to_status_id': None,
# 	u'id': 723964024451305473,
# 	u'id_str': u'723964024451305473',
# 	u'favorite_count': 0,
# 	u'entities': {
# 		u'symbols': [],
# 		u'user_mentions': [],
# 		u'hashtags': [],
# 		u'urls': []
# 	},
# 	u'retweeted': False,
# 	u'coordinates': None,
# 	u'source': u'<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
# 	u'in_reply_to_screen_name': None,
# 	u'in_reply_to_user_id': None,
# 	u'retweet_count': 0,
# 	u'favorited': False,
# 	u'user': {
# 		u'follow_request_sent': False,
# 		u'has_extended_profile': False,
# 		u'profile_use_background_image': False,
# 		u'default_profile_image': False,
# 		u'id': 68090172,
# 		u'profile_background_image_url_https':
# 		u'https://abs.twimg.com/images/themes/theme5/bg.gif',
# 		u'verified': False,
# 		u'profile_text_color': u'000000',
# 		u'profile_image_url_https': u'https://pbs.twimg.com/profile_images/723421784804974592/xrR8teIJ_normal.jpg',
# 		u'profile_sidebar_fill_color': u'000000',
# 		u'entities': {
# 			u'url': {
# 				u'urls': [{u'url': u'https://t.co/l5LrI3tOui', u'indices': [0, 23], u'expanded_url': u'https://alyzoso.tumblr.com', u'display_url': u'alyzoso.tumblr.com'}]
# 			},
# 			u'description': {u'urls': []}
# 		},
# 		u'followers_count': 462,
# 		u'profile_sidebar_border_color': u'000000',
# 		u'id_str': u'68090172',
# 		u'profile_background_color': u'000000',
# 		u'listed_count': 1,
# 		u'is_translation_enabled': False,
# 		u'utc_offset': -25200,
# 		u'statuses_count': 10826,
# 		u'description': u'You must go on living to create something to live for, even if right now there is nothing.',
# 		u'friends_count': 683,
# 		u'location': u'that green house on hoover',
# 		u'profile_link_color': u'7FDBB6',
# 		u'profile_image_url': u'http://pbs.twimg.com/profile_images/723421784804974592/xrR8teIJ_normal.jpg',
# 		u'following': False, u'geo_enabled': True,
# 		u'profile_banner_url': u'https://pbs.twimg.com/profile_banners/68090172/1436768088',
# 		u'profile_background_image_url': u'http://abs.twimg.com/images/themes/theme5/bg.gif',
# 		u'screen_name': u'wonderwall20',
# 		u'lang': u'en',
# 		u'profile_background_tile': False,
# 		u'favourites_count': 5943,
# 		u'name': u'Al.\u2652',
# 		u'notifications': False,
# 		u'url': u'https://t.co/l5LrI3tOui',
# 		u'created_at': u'Sun Aug 23 06:30:50 +0000 2009',
# 		u'contributors_enabled': False,
# 		u'time_zone': u'Pacific Time (US & Canada)',
# 		u'protected': False,
# 		u'default_profile': False,
# 		u'is_translator': False
# 	},
# 	u'geo': None,
# 	u'in_reply_to_user_id_str': None,
# 	u'lang': u'en',
# 	u'created_at': u'Sat Apr 23 19:57:28 +0000 2016',
# 	u'in_reply_to_status_id_str': None,
# 	u'place': None,
# 	u'metadata': {u'iso_language_code': u'en', u'result_type': u'recent'}
# }

if __name__ == '__main__':
	oauth = credsfromfile()
	client = Query(**oauth)
	with open(searchfile, 'rb') as f_search:
		
		search_terms = [term.strip() for term in f_search.readlines() if term.strip()]

		# Get tweets for specific search terms
		for term in search_terms:
			print "Collecting {term}".format(term=term)
			search_data = []

			tweets = client.search_tweets(keywords="{term} -filter:retweets".format(term=term), limit=float('inf'))
			while True:
				tweet = next(tweets, None)
				if tweet is None:
					break
				elif tweet['id_str'] in tweet_ids:
					continue

				author_names.add(tweet['user']['screen_name'])
				tweet_ids.add(tweet['id_str'])
				search_data.append(tweet)

				tweet['text'] = unicodedata.normalize('NFKD', strip_emoji(tweet['text']) ).encode('ascii', 'ignore').encode('utf-8')
				tweet['query'] = term

				if len(search_data) >= num_topic_tweets:
					break

			data.append(search_data)

	# Get author tweets
	# print 'Collecting author tweets'
	# author_data = []

	# sampled_author_names = list(author_names)
	# shuffle(sampled_author_names)
	# sampled_author_names = sampled_author_names[:20]
	# print sampled_author_names

	# tweets = client.search_tweets(keywords="{search} -filter:retweets".format(search=' OR '.join(["from:{0}".format(user) for user in sampled_author_names])), limit=float('inf'))
	# while True:
	# 	tweet = next(tweets, None)
	# 	if tweet is None:
	# 		break
	# 	elif tweet['id_str'] in tweet_ids:
	# 		continue

	# 	tweet_ids.add(tweet['id_str'])
	# 	author_data.append(tweet)

	# 	if len(author_data) >= num_author_tweets:
	# 		break

	# Shuffle

	for i, _ in enumerate(data):
		tweet_group = data[i]
		shuffle(tweet_group)

	# Write data to file
	with open(train_outputfile, 'wb') as f_train, open(test_outputfile, 'wb') as f_test, open(author_file, 'a') as f_author:
		# Set up author data
		# author_data = [[t['id_str'], t['created_at'], t['user']['screen_name'], t['retweeted'], ','.join(ht['text'] for ht in t['entities']['hashtags']), t['text'] ] for t in author_data]

		# Split up 80/20
		train_split = []
		test_split = []
		
		for tweet_group in data:
			train_split.append(tweet_group[:int(0.8*num_topic_tweets)])
			test_split.append(tweet_group[int(0.8*num_topic_tweets):])

		train_data = []
		for tweet_group in train_split:
			for tweet in tweet_group:
				train_data.append(tweet)

		# train_data = [tweet for tweet in tweet_group for tweet_group in train_split]

		for i, t in enumerate(train_data):
			hashtag_text = ','.join(ht['text'] for ht in t['entities']['hashtags'])
			train_data[i] = [ t['id_str'], t['created_at'], t['query'], t['user']['screen_name'], t['retweeted'], hashtag_text, t['text'] ]

		# train_data = [[t['id_str'], t['created_at'], t['user']['screen_name'], t['retweeted'], ','.join(ht['text'] for ht in t['entities']['hashtags']), t['text'] ] for t in train_data]
		# train_data = train_data + author_data[:int(0.8*num_author_tweets)]

		test_data = []
		for tweet_group in test_split:
			for tweet in tweet_group:
				test_data.append(tweet)

		# test_data = [tweet for tweet in tweet_group for tweet_group in test_split]
		
		for i, t in enumerate(test_data):
			hashtag_text = ','.join(ht['text'] for ht in t['entities']['hashtags'])
			test_data[i] = [ t['id_str'], t['created_at'], t['query'], t['user']['screen_name'], t['retweeted'], hashtag_text, t['text'] ]

		# test_data = [[t['id_str'], t['created_at'], t['user']['screen_name'], t['retweeted'], ','.join(ht['text'] for ht in t['entities']['hashtags']), t['text'] ] for t in test_data]
		# test_data = test_data + author_data[int(0.8*num_author_tweets):]

		csv.writer(f_train).writerows(train_data)
		csv.writer(f_test).writerows(test_data)

		# csv.writer(f_author).writerows(list(author_names))

