import string
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from gensim import utils

#An iterator that yields sentences (lists of str).
class corpus:
    def __iter__(self):
        with open('../gloveTut/glove/clean_sent.txt', 'rb') as file:
            for line in file:
                yield utils.simple_preprocess(line)

# split into words
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# convert to lower case
def lowerCase(tokens):
    tokens = [w.lower() for w in tokens]
    return tokens

# remove punctuation from each word
def removePunctuation(tokens):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    return stripped

# remove remaining tokens that are not alphabetic or numeric
def removeNonalphaOrNonnum(tokens):
    words = []
    for word in tokens:
        if word.isalpha() or word.isnumeric():
            words.append(word)
    return words

def main():
    clean_sentences = []

    #read in training corpus as a list of sentences
    with open('double_hard_data/temp_sentences.txt', 'rb') as file:
        lines = file.readlines()

    #creates tokens for each sentence and clean
    for line in lines:
        tokens = tokenize(line.decode('utf-8'))
        tokens = removePunctuation(tokens)
        tokens = lowerCase(tokens)
        tokens = removeNonalphaOrNonnum(tokens)
        clean_sentences.append(' '.join(tokens))

    #write each sentence token to a new line in a file
    with open('../gloveTut/glove/clean_sent.txt', 'w') as file:
        for s in clean_sentences:
            file.write(s + '\n')


if __name__ == "__main__":
    main()