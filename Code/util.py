import csv
def loadData(file_name):
	'''
		Reading tweets from a file.

		Input:
				file_name: path to the file.
		Output:
				A list with the text of the tweet:

	'''
	with open(file_name,'rb') as csvfile:
		reader = csv.reader(csvfile)
		tweets = [row[-1] for row in reader]
	return tweets
