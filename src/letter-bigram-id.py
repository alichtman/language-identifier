# alichtman

import bigrams
import string
from pprint import pprint


def create_letter_dict(file_text):
	"""
	Returns map of letter bigrams to their frequency from a list of words.
	"""

	# Remove all punctuation and new lines and spaces
	# https://stackoverflow.com/a/42025189/8740440
	removal_keys = string.punctuation

	for removal_key in removal_keys:
		file_text = file_text.replace(removal_key, "")

	letter_list = [file_text[i:i + 2] for i in range(len(file_text))]

	# pprint(letter_list)

	return bigrams.bigrams(letter_list)


def word_count(bigrams_dict):
	"""Takes a dictionary mapping words to a set of dictionaries mapping successive words to their frequencies.
	Returns a dictionary mapping words to total number of times seen. Draws counts from both the number of times
	*word* is the first part of a bigram and the second part of a bigram."""

	# Count frequency of each word in text
	word_freq_dict = {}
	for word in bigrams_dict.keys():
		if word is None:
			# Stops weird error from being thrown idk
			# print("NONE NOT SCRAPED")
			pass
		else:
			next_words_dict = bigrams_dict[word]
			if word not in word_freq_dict:
				word_freq_dict[word] = 0

			# Add number of times next_word appears after word
			for next_word in next_words_dict:
				word_freq_dict[word] += next_words_dict.get(next_word)

				if next_word not in word_freq_dict:
					word_freq_dict[next_word] = next_words_dict.get(next_word)
				else:
					word_freq_dict[next_word] += next_words_dict.get(next_word)

	# pprint(word_freq_dict)
	return word_freq_dict


def vocabulary_size(word_dict):
	"""Returns the vocabulary size of a word count dictionary"""

	return len(word_dict)


def create_probability_matrix(bigrams_matrix, w0_count, smoothing):
	"""Calculate relative probabilities of each word in the bigram matrix with optional smoothing.
	smoothing = 0 -> No smoothing
	smoothing = 1 -> LaPlace (Add 1)
	smoothing = 2 -> Good-Turing
	"""
	# print("BIGRAMS MATRIX START STATE")
	# pprint(bigrams_matrix)

	# No smoothing
	# P = C(w0w1) / C(w0)
	if smoothing == 0:
		for w0 in bigrams_matrix.keys():
			next_words = bigrams_matrix[w0]

			for w0w1_sequence in next_words.keys:
				bigrams_matrix[w0][w0w1_sequence] = (next_words.get(w0w1_sequence) / w0_count[w0])

	# LaPlace Smoothing
	# P = (w0_w1 + 1) / (w0_count + vocabulary_size)
	elif smoothing == 1:
		# print("LAPLACE")
		v = vocabulary_size(w0_count)
		# print("VOCAB SIZE:", v)
		for w0 in bigrams_matrix:
			next_words = bigrams_matrix[w0]
			# print("FIRST WORD:", w0)  # , "\tNEXT:")
			# pprint(next_words.keys())
			for w0w1_sequence in next_words:
				# print("NEXT WORD:", w0w1_sequence)
				bigrams_matrix[w0][w0w1_sequence] = (next_words[w0w1_sequence] + 1) / (w0_count[w0] + v)

	return bigrams_matrix


def calculate_error(test_matrix, model_matrix):
	"""
	Returns MSE of test matrix from model probability matrix.
	https://en.wikipedia.org/wiki/Mean_squared_error
	"""
	error = 0
	for key, values in test_matrix.items():
		mse = 0
		for key_word, val_word in values.items():
			error += (val_word - model_matrix.get(key, {}).get(key_word, 1)) ** 2
		mse += error / len(values)
	return error


def processing_pipeline(file, smoothing):
	"""
	Run full processing pipeline on file.
	"""

	# Get bigram mapping of training data
	#        word -> {(next word -> count), ...}
	letter_bigram_dict = create_letter_dict(open(file).read().replace("\n", ""))

	pprint(letter_bigram_dict)

	# Count number of times each word is seen in order to calculate probabilities
	letter_bigram_count = word_count(letter_bigram_dict)

	# Create probability matrix
	return create_probability_matrix(letter_bigram_dict, letter_bigram_count, smoothing)


def accuracy_check(results, solution):
	"""Prints accuracy of language identification model"""

	# https://stackoverflow.com/a/16289797/8740440
	with open(results) as out:
		with open(solution) as soln:
			same = set(out).intersection(soln)
			soln.seek(0)
			out.seek(0)
			diff = set(soln).difference(out)
	print("\n----------\nLanguage Identification Evaluation\n----------\n")
	print("Letter Bigrams with Add One Smoothing...")
	print("Correct: " + str(len(same)) + "\nAcc: " + str(float(str(len(same) / (len(same) + len(diff)))[0:7]) * 100) + "%")


if __name__ == "__main__":

	##############
	# Train English, French and Italian language models
	##############

	print("Training english, french and italian models...")

	# Construct probability matrices with add one smoothing
	english_prob_matrix = processing_pipeline("../data/english-train.txt", smoothing=1)
	french_prob_matrix = processing_pipeline("../data/french-train.txt", smoothing=1)
	italian_prob_matrix = processing_pipeline("../data/italian-train.txt", smoothing=1)

	print("Done training!")

	##############
	# Apply models to determine most likely language
	##############

	print("Reading test data...")

	# Look at each line individually
	line_num = 1
	with open("../data/test-input.txt", "r") as test_file:
		with open("../output/letter-bigram-out.txt", "w+") as output:
			for line in test_file:
				line_bigrams = create_letter_dict(line)
				word_freq = word_count(line_bigrams)
				line_prob_matrix = create_probability_matrix(line_bigrams, word_freq, smoothing=1)

				# Calculate error of all three models
				english_err = calculate_error(line_prob_matrix, english_prob_matrix)
				french_err = calculate_error(line_prob_matrix, french_prob_matrix)
				italian_err = calculate_error(line_prob_matrix, italian_prob_matrix)

				# Find min error
				min_err = min(english_err, french_err, italian_err)

				print("MIN ERR:", min_err)

				# Print output to file
				if min_err == english_err:
					print("ENG")
					output.write(str(line_num) + ' English\n')
				elif min_err == french_err:
					print("FRE")
					output.write(str(line_num) + ' French\n')
				elif min_err == italian_err:
					print("ITA")
					output.write(str(line_num) + ' Italian\n')

				line_num += 1

	##############
	# Evaluate model performance
	##############

	accuracy_check("../data/correct-responses.txt", "../output/letter-bigram-out.txt")
