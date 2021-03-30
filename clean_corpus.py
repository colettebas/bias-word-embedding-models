import string
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from gensim import utils

class corpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        with open('../gloveTut/glove/clean_sent.txt', 'rb') as file:
            for line in file:
                yield utils.simple_preprocess(line)

def tokenize(text):
    # split into words
    tokens = word_tokenize(text)
    return tokens

def lowerCase(tokens):
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    return tokens

def removePunctuation(tokens):
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    return stripped

def removeNonalphaOrNonnum(tokens):
    # remove remaining tokens that are not alphabetic or numeric
    words = []
    for word in tokens:
        if word.isalpha() or word.isnumeric():
            words.append(word)
    return words

def main():
    # load double_hard_data
    #filename = 'double_hard_data/polusa.txt'
    #file = open(filename, 'rt')
    #text = file.read()
    #file.close()
    #sentences = sent_tokenize(text)
    #with open('temp_sentences.txt', 'w') as file:
    #    for s in sentences:
    #        file.write(s + '\n')
    clean_sentences = []
    with open('double_hard_data/temp_sentences.txt', 'rb') as file:
        lines = file.readlines()
    for line in lines:
        tokens = tokenize(line.decode('utf-8'))
        tokens = removePunctuation(tokens)
        tokens = lowerCase(tokens)
        tokens = removeNonalphaOrNonnum(tokens)
        clean_sentences.append(' '.join(tokens))
    with open('../gloveTut/glove/clean_sent.txt', 'w') as file:
        for s in clean_sentences:
            file.write(s + '\n')


if __name__ == "__main__":
    main()