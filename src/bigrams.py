from dicts import DefaultDict


def bigrams(words):
	"""
	Given an array of words, returns a dictionary of dictionaries,
	containing occurrence counts of bigrams.
	"""
	d = DefaultDict(DefaultDict(0))
	for (w1, w2) in zip([None] + words, words + [None]):
		d[w1][w2] += 1
	return d


def file2bigrams(filename):
	return bigrams(open(filename).read().split())
