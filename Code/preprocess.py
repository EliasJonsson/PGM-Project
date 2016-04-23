from nltk.corpus import stopwords
from nltk.corpus import wordnet
from string import punctuation
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
import re

def removeEllipsis(tweets):
	'''
	Remove all ellipsis, like ..... ,..,.,. !!.., etc., from the tweets.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without ellipsis 

	'''
	punct = r'[{}]'.format(re.escape(punctuation))
	punct = punct + r'{2,}'
	tweets = [re.sub(punct, "", t) for t in tweets]
	return tweets

def removeUsers(tweets):
	'''
	Remove twitter users, like @jess, @user_1 etc.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without users. 

	'''

	pat = r'((?<=\W)|^)@\S*'
	tweets = [re.sub(pat, "", t) for t in tweets]
	return tweets


def removeClitics(tweets):
	'''
	Remove clitics. Don't, we'll, etc.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets with no clitics.
	'''
	pat = r'(^|(?<=\s))\S*[\'`]\S*($|(?=\s))'
	tweets = [re.sub(pat, "", t) for t in tweets]
	
	return tweets

def removePunctuations(tweets):
	'''
	Erase all punctuations that are appended to a word, prepended to a word or stands alone.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without punctuations that are appended to a word, prepended to a word, or stands alone.

	'''
	punct = r'[\.,;:\(\)\[\]\?!#]'

	# Erase all punctuations that are appended to a word.
	pat = r'((?<=\w)|^)' + punct + r'(?=(\s|$))'
	tweets = [re.sub(pat, "", t) for t in tweets]

	# Erase all punctuations that are prepended to a word.
	pat = r'((?<=\s)|^)' + punct + r'(?=(\w|$))'
	tweets = [re.sub(pat, "", t) for t in tweets]

	# Erase all punctuations that stands alone.
	pat = r'((?<=\s)|^)' + punct + r'((?=\s)|$)'
	tweets = [re.sub(pat, "", t) for t in tweets]

	return tweets

def removeHTMLTags(tweets):
	'''
	Remove all html tags such as <ref>

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without html tags.
	'''
	pat = r'<\S*>'
	tweets = [re.sub(pat,'', t) for t in tweets]
	return tweets

def removeURL(tweets):
	'''
	Remove all url

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without url.
	'''
	pat = r'(www\.|http)\S*\s'
	tweets = [re.sub(pat, '', t) for t in tweets];
	return tweets


def removePunctuationWords(tweets):
	'''
	Remove all words that includes punctuations, except abbreviations.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without words with punctuations except abbreviations.
	'''
	punct = r'[{}]'.format(re.escape(punctuation))
	punct_exception = re.sub(r'[\.#]','',punct)
	pat = r'(^|\s)\S*' + punct_exception + r'\S*($|\s)'
	tweets = [re.sub(pat, ' ', t) for t in tweets];
	return tweets



def removeShortWords(tweets):
	'''
	Remove all single or two letter words.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without single or two letter words.
	'''
	pat = r'(^|\s)\S{1,2}(\s|$)'
	tweets = [re.sub(pat, ' ', t) for t in tweets];
	return tweets
	
def lowercase(tweets):
	'''
	Let all words only include lowercase letters.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets only with lowercase letters.
	'''
	tweets = [t.lower() for t in tweets]
	return tweets

def removeExtraSpaces(tweets):
	'''
	Remove all extra spaces

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without extra spaces.
	'''

	# Erase all double spaces.
	pat = r'\s{2,}'
	tweets = [re.sub(pat, " ", t) for t in tweets]

	# Erase all spaces in the begining or in the end of a tweet.
	pat = r'\s$|^\s'
	tweets = [re.sub(pat, "", t) for t in tweets]

	return tweets

def splitWords(tweets):
	'''
	Split all the tweets up in words.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets:List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   	  [word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						      [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]

	'''
	tweets = [t.split() for t in tweets]
	return tweets

def removeStopwords(tweets):
	'''
	Remove all stopwords

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets without stopwords.
	'''

	stop = stopwords.words('english')
	stop = [s.lower() for s in stop]
	tweets = [[w for w in t if w not in set(stop)] for t in tweets]
	#tweets = [list(set(t)-set(stop)) for t in tweets]
	return tweets

def lemmatize(tweets):
	'''
	Lemmatize words in the corpus.
	
	Input:
	------------------
	tweets: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   	[word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						    [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]
	Output:
	-----------------
	newTweets: All the words in the tweet lemmatized.
	'''
	wordnet_lemmatizer = WordNetLemmatizer()
	pos_tag_tweets = [nltk.pos_tag(t) for t in tweets]
	tweets = []
	i = 0
	for t in pos_tag_tweets:
		tt = []
		for w in t:
			if get_wordnet_pos(w[1]) =='':
				tt.append(w[0])
			else:
				try:
					tt.append(wordnet_lemmatizer.lemmatize(w[0], pos = get_wordnet_pos(w[1])))
				except UnicodeDecodeError:
					pass
		tweets.append(tt)
		i += 1
		print i
	return tweets


def remove_infrequent_words(tweets):
	'''
	Remove all words that appear in less than 10 documents

	Input:
	------------------
	tweets: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   	[word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						    [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]

	Output:
	-----------------
	newTweets: All the words in the tweet except the words that appear in less than 10 documents.
	'''
	cap = 1
	df = Counter()
	for t in tweets:
		df += Counter(set(t))
	print df
	tweets = [[w for w in t if df[w]>cap]for t in tweets]
	return tweets

def remove_short_tweets(tweets):
	'''
	Remove tweets with less than or equal to two words

	Input:
	------------------
	tweets: List of lists, [[word1OfTweet1, word2OfTweet1,...,word_m1OfTweet1],
						   	[word1OfTweet2, word2OfTweet2,...,word_m2OfTweet2],
						   						. 								
						   						. 
						   						.
						    [word1OfTweetN, word2OfTweetN,...,word_mNOfTweetN]]

	Output:
	-----------------
	newTweets: All the tweets in tweets except all tweets shorter or equal to two words are removed.
	'''	
	tweets = [t for t in tweets if len(t) > 2]
	return tweets

def preprocess(tweets):
	'''
	Preprocessing for topic models.

	Input:
	------------------
	tweets: A list of tweets. [tweet 1, tweet2, ..., tweetN]

	Output:
	-----------------
	newTweets: tweets ready for the training of topic models.
	'''
	tweets = removeEllipsis(tweets)
	tweets = removeUsers(tweets)
	tweets = removeHTMLTags(tweets)
	tweets = removeURL(tweets)
	tweets = removeClitics(tweets)
	tweets = removePunctuations(tweets)
	tweets = removePunctuationWords(tweets)
	tweets = removeShortWords(tweets)
	tweets = lowercase(tweets)
	tweets = removeExtraSpaces(tweets)
	tweets = splitWords(tweets)
	tweets = removeStopwords(tweets)
	tweets = lemmatize(tweets)
	tweets = remove_infrequent_words(tweets)
	tweets = remove_short_tweets(tweets)
	
	return tweets








def get_wordnet_pos(tag):
	'''
	Change tag to a normal nltk lemmatize inputs.

	Input:
	-----------------
	tag: Nltk tag, from Nltk tag of speech function.

	Output:
	-----------------
	c: proper nltk lemmatize input word for a tag.
	'''
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return ''


	










	
