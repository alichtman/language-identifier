# alichtman

from bigrams import convert_file_to_bigrams, get_bigrams
import string


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

	return get_bigrams(letter_list)


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

			for w0w1_sequence in next_words.keys():
				bigrams_matrix[w0][w0w1_sequence] = (next_words.get(w0w1_sequence) / w0_count[w0])

	# LaPlace Smoothing
	# P = (w0_w1 + 1) / (w0_count + vocabulary_size)
	elif smoothing == 1:
		# print("LAPLACE")
		v = vocabulary_size(w0_count)
		for w0 in bigrams_matrix:
			next_words = bigrams_matrix[w0]
			# print("FIRST WORD:", w0)  # , "\tNEXT:")
			# pprint(next_words.keys())
			for w0w1_sequence in next_words:
				bigrams_matrix[w0][w0w1_sequence] = (next_words[w0w1_sequence] + 1) / (w0_count[w0] + v)

	# Good-Turing Smoothing
	#
	# v = Total number of words
	# N_count = Number of words seen count times
	#
	# P_GT (zero) = N_1 / v
	# P_GT (non-zero) = ( (count + 1) * (N_count+1) / (N_count) ) / v
	elif smoothing == 2:
		print("GOOD TURING")
		v = vocabulary_size(w0_count)
		print("Vocab Size:", v)
		for w0 in bigrams_matrix:
			next_words = bigrams_matrix[w0]
			print("FIRST WORD:", w0)  # , "\tNEXT:")
			# pprint(next_words.keys())
			for w0w1_sequence in next_words:
				# P_GT (zero) = N_1 / N
				if bigrams_matrix[w0][w0w1_sequence] == 0:
					# bigrams_matrix[w0][w0w1_sequence] = N_1 / v
					bigrams_matrix[w0][w0w1_sequence] = (next_words[w0w1_sequence] + 1) / v
				# P_GT (non-zero) = ( (count + 1) * (N_count+1) / (N_count) ) / v
				else:
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


def bigram_processing(bigram_type, file, smoothing):
	"""
	Run word processing pipeline on file.
	"""
	if bigram_type == "word":
		# Get bigram word mapping of training data
		#        word -> {(next word -> count), ...}
		bigram_dict = convert_file_to_bigrams(file)
	else:
		bigram_dict = create_letter_dict(open(file).read().replace("\n", ""))

	# Count number of times each word is seen in order to calculate probabilities
	individual_word_count = word_count(bigram_dict)

	# Create probability matrix
	probability_matrix = create_probability_matrix(bigram_dict, individual_word_count, smoothing)
	return probability_matrix


def test_models(output_file, word_bigram, eng_model, fre_model, ita_model, smoothing):
	"""
	Iterate through test file and test all models against each line.
	Assign the language with the least error to the corresponding line
	in the output file.

	:@param: word_bigram 	True if it's a word bigram model, False if letter.
	:@param: smoothing 		0 for No Smoothing, 1 for LaPlace, 2 for Good Turing
	"""

	# Look at each line individually
	line_num = 1
	with open("../data/test-input.txt", "r") as test_file:
		with open(output_file, "w+") as output:
			for line in test_file:
				# Get word or letter bigrams
				if word_bigram:
					line_bigrams = get_bigrams(line.split())
				else:
					line_bigrams = create_letter_dict(line)

				word_freq = word_count(line_bigrams)
				line_prob_matrix = create_probability_matrix(line_bigrams, word_freq, smoothing=1)

				# Calculate error of all three models
				english_err = calculate_error(line_prob_matrix, eng_model)
				french_err = calculate_error(line_prob_matrix, fre_model)
				italian_err = calculate_error(line_prob_matrix, ita_model)

				# Find min error
				min_err = min(english_err, french_err, italian_err)

				# print("MIN ERR:", min_err)

				# Add predicted language with line number to file
				if min_err == english_err:
					# print("ENG")
					output.write(str(line_num) + ' English\n')
				elif min_err == french_err:
					# print("FRE")
					output.write(str(line_num) + ' French\n')
				elif min_err == italian_err:
					# print("ITA")
					output.write(str(line_num) + ' Italian\n')

				line_num += 1


def accuracy_check(predictions, solutions):
	"""
	Calculates and prints accuracy of language identification model.
	"""
	# https://stackoverflow.com/a/16289797/8740440
	with open(predictions) as out:
		with open(solutions) as soln:
			same = set(out).intersection(soln)
			soln.seek(0)
			out.seek(0)
			diff = set(soln).difference(out)

	# First word in path
	model_type = predictions.split("-")[0].split("/")[-1].capitalize()
	# Third word from end of path
	smoothing = predictions.split("-")[-3].capitalize()

	print("\n{} Bigrams with {} Smoothing...".format(model_type, smoothing))
	print("\tCorrect Predictions: " + str(len(same)) + "\n\tTotal Predictions: " + str(
		len(same) + len(diff)) + "\n\tAccuracy: " + str(
		float(str(len(same) / (len(same) + len(diff)))[0:7]) * 100) + "%")


def main():
	print("Training english, french and italian models...")

	##############
	# Train letter bigram models with no smoothing
	##############

	print("\tTraining letter bigram models with no smoothing...")

	eng_letter_bigrams_no_smoothing = bigram_processing("letter", "../data/english-training.txt", smoothing=0)
	fre_letter_bigrams_no_smoothing = bigram_processing("letter", "../data/french-training.txt", smoothing=0)
	ita_letter_bigrams_no_smoothing = bigram_processing("letter", "../data/italian-training.txt", smoothing=0)

	##############
	# Train letter bigram models with add one smoothing
	##############

	print("\tTraining letter bigram models with LaPlace smoothing...")

	eng_letter_bigrams_laplace = bigram_processing("letter", "../data/english-training.txt", smoothing=1)
	fre_letter_bigrams_laplace = bigram_processing("letter", "../data/french-training.txt", smoothing=1)
	ita_letter_bigrams_laplace = bigram_processing("letter", "../data/italian-training.txt", smoothing=1)

	##############
	# Train word bigram models with no smoothing
	##############

	print("\tTraining word bigram models with no smoothing...")

	eng_word_bigrams_no_smoothing = bigram_processing("word", "../data/english-training.txt", smoothing=0)
	fre_word_bigrams_no_smoothing = bigram_processing("word", "../data/french-training.txt", smoothing=0)
	ita_word_bigrams_no_smoothing = bigram_processing("word", "../data/italian-training.txt", smoothing=0)

	##############
	# Train word bigram models with add one smoothing
	##############

	print("\tTraining word bigram models with LaPlace smoothing...")

	eng_word_bigrams_laplace = bigram_processing("word", "../data/english-training.txt", smoothing=1)
	fre_word_bigrams_laplace = bigram_processing("word", "../data/french-training.txt", smoothing=1)
	ita_word_bigrams_laplace = bigram_processing("word", "../data/italian-training.txt", smoothing=1)

	print("-> Done training!\n")

	##############
	# Apply models to determine most likely language
	##############

	print("Appling each model to create a set of language predictions...")

	# Letter bigram models

	print("\tLetter bigram models with no smoothing...")
	test_models("../output/letter-bigram-no-smoothing-predictions.txt",
	            False,
	            eng_letter_bigrams_no_smoothing,
	            fre_letter_bigrams_no_smoothing,
	            ita_letter_bigrams_no_smoothing,
	            0)

	print("\tLetter bigram models with LaPlace smoothing...")
	test_models("../output/letter-bigram-laplace-smoothing-predictions.txt",
	            False,
	            eng_letter_bigrams_laplace,
	            fre_letter_bigrams_laplace,
	            ita_letter_bigrams_laplace,
	            1)

	# Word bigram models

	print("\tWord bigram models with no smoothing...")
	test_models("../output/word-bigram-no-smoothing-predictions.txt",
	            True,
	            eng_word_bigrams_no_smoothing,
	            fre_word_bigrams_no_smoothing,
	            ita_word_bigrams_no_smoothing,
	            0)

	print("\tWord bigram models with LaPlace smoothing...")
	test_models("../output/word-bigram-laplace-smoothing-predictions.txt",
	            True,
	            eng_word_bigrams_laplace,
	            fre_word_bigrams_laplace,
	            ita_word_bigrams_laplace,
	            1)

	##############
	# Evaluate model performance
	##############

	print("\n##################\n# Model Evaluation\n##################")

	accuracy_check("../output/letter-bigram-no-smoothing-predictions.txt",
	               "../data/correct-responses.txt")

	accuracy_check("../output/letter-bigram-laplace-smoothing-predictions.txt",
	               "../data/correct-responses.txt")

	accuracy_check("../output/word-bigram-no-smoothing-predictions.txt",
	               "../data/correct-responses.txt")

	accuracy_check("../output/word-bigram-laplace-smoothing-predictions.txt",
	               "../data/correct-responses.txt")

	print("\nDiff the output files to see which lines were predicted differently by certain pairs of models.")
	print("Here are some commands to try:")
	print(
		"\n$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/letter-bigram-no-smoothing-predictions.txt")
	print(
		"$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/word-bigram-no-smoothing-predictions.txt")
	print(
		"$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/word-bigram-laplace-smoothing-predictions.txt")


if __name__ == "__main__":
	main()
