import load_vectors


def main():
    return


# return the top 20 most similar word vectors in the vocabulary
def most_similar(model, target):
    return model.most_similar(positive = target, topn=20)


# save word2vec and fastText models as text files with the word followed by the vector on each line
def convert_to_w2v_format(w2v_in, fast_in, w2v_name, fast_name):
    print('loading fastText')
    fast_text_model = load_vectors.load_fast_text(fast_in)
    print('loading w2v')
    w2v_model = load_vectors.load_w2v(w2v_in)
    fast_text_model.wv.save_word2vec_format(fast_name)
    w2v_model.wv.save_word2vec_format(w2v_name)


# given two models, one as a text files and one Gensim word2vec, identify different words in each vocabulary
def id_diff_vocab():
    print('loading glove')
    glove_model = load_vectors.load_txt_format('glove_vectors.txt')
    print('loading w2v')
    w2v_model = load_vectors.load_w2v('w2v_vectors')
    glove_vocab = list(glove_model.vocab)
    w2v_vocab = list(w2v_model.wv.vocab)

    # find the difference between the vocabularies
    diff_vocab = set(glove_vocab).symmetric_difference(w2v_vocab)

    # write glove vocabulary to a text file
    with open('glove_vocab_from_model.txt', 'w') as file:
        for v in glove_vocab:
            file.write(v + '\n')

    # write word2vec vocabulary to a text file
    with open('w2v_vocab_from_model.txt', 'w') as file:
        for v in w2v_vocab:
            file.write(v + '\n')

    # write different words to a text file
    with open('vocab_diff.txt', 'w') as file:
        for v in diff_vocab:
            file.write(v + '\n')

if __name__ == "__main__":
    main()