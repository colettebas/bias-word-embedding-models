import json

import load_vectors
from responsibly.we import calc_all_weat, calc_single_weat

def calc_benchmark_scores(model, f_name):
    with open('benchmark_weat.json') as data:
        dict = json.load(data)
    result = calc_all_weat(model, weat_data = dict, with_original_finding=True, with_pvalue=False)
    result.to_csv(f_name)

def calc_custom_scores(model, f_name):
    with open('custom_weat.json') as data:
        dict = json.load(data)
    result = calc_all_weat(model, weat_data = dict, with_original_finding=False, with_pvalue=False)
    result.to_csv(f_name)

def main():
  #  #load w2v
  #  w2v1_model = load_vectors.load_w2v('w2v_vectors')
  #  w2v2_model = load_vectors.load_w2v('w2v_vectors2')
  #  w2v3_model = load_vectors.load_w2v('w2v_vectors3')
  #  w2v4_model = load_vectors.load_w2v('w2v_vectors4')
  #  w2v5_model = load_vectors.load_w2v('w2v_vectors5')
  #  print('done')
#
  #  #load glove
  #  glove1_model = load_vectors.load_glove('glove_vectors.txt')
  #  glove2_model = load_vectors.load_glove('glove_vectors2.txt')
  #  glove3_model = load_vectors.load_glove('glove_vectors3.txt')
  #  glove4_model = load_vectors.load_glove('glove_vectors4.txt')
  #  glove5_model = load_vectors.load_glove('glove_vectors5.txt')
  #  print('done')

    #load fastText
    fast_text1_model = load_vectors.load_fast_text('fast_text_vectors')
    fast_text2_model = load_vectors.load_fast_text('fast_text_vectors2')
   # fast_text3_model = load_vectors.load_fast_text('fast_text_vectors3')
   # fast_text4_model = load_vectors.load_fast_text('fast_text_vectors4')
   # fast_text5_model = load_vectors.load_fast_text('fast_text_vectors5')
   # print('done')

   # calc_benchmark_scores(w2v1_model.wv, 'weat_results/w2v1_benchmark_weat.csv')
   # calc_benchmark_scores(w2v2_model.wv, 'weat_results/w2v2_benchmark_weat.csv')
   # calc_benchmark_scores(w2v3_model.wv, 'weat_results/w2v3_benchmark_weat.csv')
   # calc_benchmark_scores(w2v4_model.wv, 'weat_results/w2v4_benchmark_weat.csv')
   # calc_benchmark_scores(w2v5_model.wv, 'weat_results/w2v5_benchmark_weat.csv')
   # print('done')
#
   # calc_benchmark_scores(glove1_model, 'weat_results/glove1_benchmark_weat.csv')
   # calc_benchmark_scores(glove2_model, 'weat_results/glove2_benchmark_weat.csv')
   # calc_benchmark_scores(glove3_model, 'weat_results/glove3_benchmark_weat.csv')
   # calc_benchmark_scores(glove4_model, 'weat_results/glove4_benchmark_weat.csv')
   # calc_benchmark_scores(glove5_model, 'weat_results/glove5_benchmark_weat.csv')
   # print('done')

   # calc_benchmark_scores(fast_text1_model.wv, 'weat_results/fast_text1_benchmark_weat.csv')
   # calc_benchmark_scores(fast_text2_model.wv, 'weat_results/fast_text2_benchmark_weat.csv')
   # calc_benchmark_scores(fast_text3_model.wv, 'weat_results/fast_text3_benchmark_weat.csv')
   # calc_benchmark_scores(fast_text4_model.wv, 'weat_results/fast_text4_benchmark_weat.csv')
   # calc_benchmark_scores(fast_text5_model.wv, 'weat_results/fast_text5_benchmark_weat.csv')
   # print('done')

 #   calc_custom_scores(w2v1_model.wv, 'weat_results/w2v1_custom_weat.csv')
 #   calc_custom_scores(w2v2_model.wv, 'weat_results/w2v2_custom_weat.csv')
 #   calc_custom_scores(w2v3_model.wv, 'weat_results/w2v3_custom_weat.csv')
 #   calc_custom_scores(w2v4_model.wv, 'weat_results/w2v4_custom_weat.csv')
 #   calc_custom_scores(w2v5_model.wv, 'weat_results/w2v5_custom_weat.csv')
 #   print('done')
#
 #   calc_custom_scores(glove1_model, 'weat_results/glove1_custom_weat.csv')
 #   calc_custom_scores(glove2_model, 'weat_results/glove2_custom_weat.csv')
 #   calc_custom_scores(glove3_model, 'weat_results/glove3_custom_weat.csv')
 #   calc_custom_scores(glove4_model, 'weat_results/glove4_custom_weat.csv')
 #   calc_custom_scores(glove5_model, 'weat_results/glove5_custom_weat.csv')
 #   print('done')

    calc_custom_scores(fast_text1_model.wv, 'weat_results/fast_text1_custom_weat.csv')
    calc_custom_scores(fast_text2_model.wv, 'weat_results/fast_text2_custom_weat.csv')
   # calc_custom_scores(fast_text3_model.wv, 'weat_results/fast_text3_custom_weat.csv')
   # calc_custom_scores(fast_text4_model.wv, 'weat_results/fast_text4_custom_weat.csv')
   # calc_custom_scores(fast_text5_model.wv, 'weat_results/fast_text5_custom_weat.csv')
   # print('done')

if __name__ == "__main__":
    main()