from nltk.corpus import stopwords
from string import punctuation
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
	punct = r'[\.,;:\(\)\[\]\?!]'

	# Erase all punctuations that are appended to a word.
	pat = r'((?<=\w)|^)' + punct + r'(?=(\s|$))'
	tweets = [re.sub(pat, "", t) for t in tweets]

	# Erase all punctuations that are appended to a word.
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
	print stop
	tweets = [list(set(t)-set(stop)) for t in tweets]
	return tweets





	










	
