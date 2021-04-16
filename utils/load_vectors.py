from gensim.models import KeyedVectors, Word2Vec, FastText

# load word vectors from text files with the word followed by the vector on each line
def load_txt_format(file):
    return KeyedVectors.load_word2vec_format(file, binary=False)

# load word vectors from Gensim files for word2vec
def load_w2v(file):
    return Word2Vec.load(file)

# load word vectors from Gensim files for fastText
def load_fast_text(file):
    return FastText.load(file)