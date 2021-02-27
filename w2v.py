from gensim.models.word2vec import Word2Vec
from clean_corpus import corpus
from callback_class import callback_log
import datetime

def create_w2v(file):
    model = Word2Vec(corpus_file=file, size=300, window=6, min_count=10, workers=8, negative=5, sg=1,
                     iter=10, callbacks=[callback_log()])
    wv = model.wv
    for index, word in enumerate(wv.index2word):
        if index == 10:
            break
        print(f"word #{index}/{len(wv.index2word)} is {word}")
    return model

def main():
    corpus_file = 'data/clean_sent.txt'
    w2v_model = create_w2v(corpus_file)
    w2v_model.save('w2v_vectors5')

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(end_time - start_time)