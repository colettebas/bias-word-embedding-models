from gensim.models.fasttext import FastText
from clean_corpus import corpus
import datetime
from callback_class import callback_log

def create_fast_text(file):
    fast_model = FastText(size=300, window=6, min_count=10, workers=8, negative=5, iter=10)
    fast_model.build_vocab(corpus_file=file)
    fast_model.train(corpus_file=file, epochs=fast_model.epochs, callbacks=[callback_log()],
                     total_examples=fast_model.corpus_count, total_words=fast_model.corpus_total_words)

    for index, word in enumerate(fast_model.wv.index2word):
        if index == 10:
            break
        print(f"word #{index}/{len(fast_model.wv.index2word)} is {word}")

    return fast_model

def main():
    fast_text_model = create_fast_text('double_hard_data/clean_sent.txt')
    fast_text_model.save('fast_text_vectors5')

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(end_time - start_time)