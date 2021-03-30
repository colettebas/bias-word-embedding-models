import datetime
import json

import load_vectors
from responsibly.we import calc_all_weat, calc_single_weat


def calc_benchmark_scores(model, f_name):
    with open('benchmark_weat.json') as data:
        dict = json.load(data)
    result = calc_all_weat(model, weat_data = dict, with_original_finding=True, with_pvalue=True)
    result.to_csv(f_name)


def calc_custom_scores(model, f_name):
    with open('custom_weat.json') as data:
        dict = json.load(data)
    result = calc_all_weat(model, weat_data = dict, with_original_finding=False, with_pvalue=True)
    result.to_csv(f_name)




def main():
    start_time = datetime.datetime.now()
    #load w2v
    w2v1_model = load_vectors.load_txt_format('embeddings/Original/w2v/w2v1.txt')
    w2v2_model = load_vectors.load_txt_format('embeddings/Original/w2v/w2v2.txt')
    w2v3_model = load_vectors.load_txt_format('embeddings/Original/w2v/w2v3.txt')
    w2v4_model = load_vectors.load_txt_format('embeddings/Original/w2v/w2v4.txt')
    w2v5_model = load_vectors.load_txt_format('embeddings/Original/w2v/w2v5.txt')
    print('done')

    #load glove
    glove1_model = load_vectors.load_txt_format('embeddings/Original/GloVe/glove_vectors1.txt')
    glove2_model = load_vectors.load_txt_format('embeddings/Original/GloVe/glove_vectors2.txt')
    glove3_model = load_vectors.load_txt_format('embeddings/Original/GloVe/glove_vectors3.txt')
    glove4_model = load_vectors.load_txt_format('embeddings/Original/GloVe/glove_vectors4.txt')
    glove5_model = load_vectors.load_txt_format('embeddings/Original/GloVe/glove_vectors5.txt')
    print('done')

    #load fastText
    fast_text1_model = load_vectors.load_txt_format('embeddings/Original/fastText/fastText1.txt')
    fast_text2_model = load_vectors.load_txt_format('embeddings/Original/fastText/fastText2.txt')
    fast_text3_model = load_vectors.load_txt_format('embeddings/Original/fastText/fastText3.txt')
    fast_text4_model = load_vectors.load_txt_format('embeddings/Original/fastText/fastText4.txt')
    fast_text5_model = load_vectors.load_txt_format('embeddings/Original/fastText/fastText5.txt')
    print('done')

    calc_benchmark_scores(w2v1_model.wv, 'weat_results/original/individual/w2v1_benchmark_weat.csv')
    calc_benchmark_scores(w2v2_model.wv, 'weat_results/original/individual/w2v2_benchmark_weat.csv')
    calc_benchmark_scores(w2v3_model.wv, 'weat_results/original/individual/w2v3_benchmark_weat.csv')
    calc_benchmark_scores(w2v4_model.wv, 'weat_results/original/individual/w2v4_benchmark_weat.csv')
    calc_benchmark_scores(w2v5_model.wv, 'weat_results/original/individual/w2v5_benchmark_weat.csv')
    print('done')

    calc_benchmark_scores(glove1_model, 'weat_results/original/individual/glove1_benchmark_weat.csv')
    calc_benchmark_scores(glove2_model, 'weat_results/original/individual/glove2_benchmark_weat.csv')
    calc_benchmark_scores(glove3_model, 'weat_results/original/individual/glove3_benchmark_weat.csv')
    calc_benchmark_scores(glove4_model, 'weat_results/original/individual/glove4_benchmark_weat.csv')
    calc_benchmark_scores(glove5_model, 'weat_results/original/individual/glove5_benchmark_weat.csv')
    print('done')

    calc_benchmark_scores(fast_text1_model.wv, 'weat_results/original/individual/fast_text1_benchmark_weat.csv')
    calc_benchmark_scores(fast_text2_model.wv, 'weat_results/original/individual/fast_text2_benchmark_weat.csv')
    calc_benchmark_scores(fast_text3_model.wv, 'weat_results/original/individual/fast_text3_benchmark_weat.csv')
    calc_benchmark_scores(fast_text4_model.wv, 'weat_results/original/individual/fast_text4_benchmark_weat.csv')
    calc_benchmark_scores(fast_text5_model.wv, 'weat_results/original/individual/fast_text5_benchmark_weat.csv')
    print('done')

 #   calc_custom_scores(w2v1_model.wv, 'weat_results/debiased/individual/w2v1debiased_custom_weat.csv')
 #   calc_custom_scores(w2v2_model.wv, 'weat_results/debiased/individual/w2v2debiased_custom_weat.csv')
 #   calc_custom_scores(w2v3_model.wv, 'weat_results/debiased/individual/w2v3debiased_custom_weat.csv')
 #   calc_custom_scores(w2v4_model.wv, 'weat_results/debiased/individual/w2v4debiased_custom_weat.csv')
 #   calc_custom_scores(w2v5_model.wv, 'weat_results/debiased/individual/w2v5debiased_custom_weat.csv')
 #   print('done')
#
 #   calc_custom_scores(glove1_model, 'weat_results/debiased/individual/glove1debiased_custom_weat.csv')
 #   calc_custom_scores(glove2_model, 'weat_results/debiased/individual/glove2debiased_custom_weat.csv')
 #   calc_custom_scores(glove3_model, 'weat_results/debiased/individual/glove3debiased_custom_weat.csv')
 #   calc_custom_scores(glove4_model, 'weat_results/debiased/individual/glove4debiased_custom_weat.csv')
 #   calc_custom_scores(glove5_model, 'weat_results/debiased/individual/glove5debiased_custom_weat.csv')
 #   print('done')
#
 #   calc_custom_scores(fast_text1_model.wv, 'weat_results/debiased/individual/fastText1debiased_custom_weat.csv')
 #   calc_custom_scores(fast_text2_model.wv, 'weat_results/debiased/individual/fastText2debiased_custom_weat.csv')
 #   calc_custom_scores(fast_text3_model.wv, 'weat_results/debiased/individual/fastText3debiased_custom_weat.csv')
 #   calc_custom_scores(fast_text4_model.wv, 'weat_results/debiased/individual/fastText4debiased_custom_weat.csv')
 #   calc_custom_scores(fast_text5_model.wv, 'weat_results/debiased/individual/fastText5debiased_custom_weat.csv')
 #   print('done')

if __name__ == "__main__":
    main()