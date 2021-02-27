from gensim.models import KeyedVectors, Word2Vec, FastText

def load_glove(file):
    return KeyedVectors.load_word2vec_format(file, binary=False)

def load_w2v(file):
    return Word2Vec.load(file)

def load_fast_text(file):
    return FastText.load(file)