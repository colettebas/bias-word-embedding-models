import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap


# create graphs for the mean scores for all word evaluation tasks by category
def plot_eval_scores(file, outlier_file, category_file, similarity_file,
                     analogy_file, eval_mean_output, eval_std_output):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(file)

    # create plot
    sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = ['8-8-8 OPP', '8-8-8 Accuracy'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(outlier_file)

    plt.clf()

    #Create graph for concept categorization tasks
    sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = ['AP', 'BLESS', 'Battig', 'ESSLI_2c', 'ESSLI_2b', 'ESSLI_1a'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(category_file)
    plt.clf()

    #Create graph for word similarity tasks
    sns.barplot(x = 'Test', y = 'Score',hue = 'Model', data = scores,
                order = ['MEN', 'WS353', 'WS353R', 'WS353S', 'SimLex999', 'RW', 'RG65', 'MTurk'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(similarity_file)
    plt.clf()

    #Create graph for word analogy tasks
    sns.barplot(x = 'Test', y = 'Score', hue = 'Model',data = scores,
                order = ['Google', 'MSR', 'SemEval2012_2'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(analogy_file)

    #save mean scores and standard deviations to files
    scores.groupby(['Test', 'Model']).mean()['Score'].to_csv(eval_mean_output)
    scores.groupby(['Test', 'Model']).std()['Score'].to_csv(eval_std_output)

# create graphs for the WEAT scores for all benchmark cases
def plot_WEAT_benchmark_scores(input, graph_output, mean_output, std_output, pvalue_mean_output):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(input)

    # create plot for benchmark cases
    plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    labels = ['Flowers/Insects & Pleasant/ Unpleasant', 'Instruments/ Weapons & Pleasant/ Unpleasant',
              'European Names/African Names & Pleasant/ Unpleasant', 'Male/Female & Career/Family',
              'Math/Arts & Male/Female', 'Science/Arts & Male/Female',
              'Mental Disease/ Physical Disease & Temporary/ Permanent',
              'Young Names/Old Names & Pleasant/ Unpleasant']
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    plot.set_xticklabels(labels,
                            rotation=90)
    plot.tick_params(axis='x', which='major', pad=10)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(graph_output)

    plt.clf()

    #save mean scores and standard deviations to files
    scores.groupby(['Test', 'Model']).mean()['Score'].to_csv(mean_output)
    scores.groupby(['Test', 'Model']).std()['Score'].to_csv(std_output)
    scores.groupby(['Test', 'Model']).mean()['pValue'].to_csv(pvalue_mean_output)

# create graphs for the WEAT scores for all political cases by categories
def plot_WEAT_political_scores(input, output_graph1, output_graph2, mean_output, std_output, pvalue_mean_output,
                               pvalue_std_output):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(input)

    # create plot for generic political cases
    labels = ['Unpleasant vs. Pleasant', 'Lazy vs. Hard Working', 'Evil vs. Good',
              'Family vs. Career', 'Female names vs. Male names', 'Female terms vs. Male terms',
              'STEM vs. Arts', 'Rich vs. Poor', 'Old people’s names vs. Young people’s names']
    generic_plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = labels,
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    generic_plot.set_xticklabels(labels,
                                 rotation=90)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(output_graph1)

    plt.clf()

    # create plot for specific political cases
    labels = ['Liberal vs. Conservative', 'Climate Change Activist vs. Climate Change Denier',
              'Prochoice vs. Prolife', 'Peace vs. War', 'Pro-Immigration vs. Anti-Immigration',
              'Trust vs. Deceptive', 'Progressive vs. Moderate', 'Transparent vs. Secretive',
              'Atheist vs. Evangelical', 'Gun Control vs. Gun Rights',
              'Northern States vs. Southern States', 'Fair vs. Unfair',
              'Trickle Up Economics vs. Trickle Down Economics']
    specific_plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = labels,
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    specific_plot.set_xticklabels(labels,
                        rotation=90)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(output_graph2)

    plt.clf()

    #save mean scores and standard deviations to files
    scores.groupby(['Test', 'Model']).mean()['Score'].to_csv(mean_output)
    scores.groupby(['Test', 'Model']).std()['Score'].to_csv(std_output)
    scores.groupby(['Test', 'Model']).mean()['pValue'].to_csv(pvalue_mean_output)
    scores.groupby(['Test', 'Model']).std()['pValue'].to_csv(pvalue_std_output)

# create graph for a specific word evaluation tasks including all models
def plot_specific_eval_comparison(file, test, out_file):
    sns.set_context('paper')

    # load dataset
    all_scores = pd.read_csv(file)

    #filter by test
    scores = all_scores[all_scores.Test == test]

    labels = ['Original w2v','Debiased w2v', 'Original fastText',  'Debiased fastText', 'Original GloVe',
               'Debiased GloVe']

    # create plot
    plot = sns.barplot(x = 'Embedding', y = 'Score', hue = 'Embedding', data = scores,
                order = labels,
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                dodge=False
                )
    plt.ylim(0.6, 0.8)
    labels = [ '\n'.join(wrap(l, 8)) for l in labels ]
    plot.set_xticklabels(labels, rotation=45)
    plot.get_legend().remove()
    plt.tight_layout()
    plt.savefig(out_file)

    plt.clf()

# create graphs for all word evaluation tasks for one model
def plot_eval_comparison(file, model, debiased_model, outlier_file, category_file, similarity_file, analogy_file):
    sns.set_context('paper')

    # load dataset
    all_scores = pd.read_csv(file)

    #filter by model
    original = all_scores[all_scores.Embedding == model]
    debiased = all_scores[all_scores.Embedding == debiased_model]
    scores = pd.concat([original, debiased])

    # create plot for outlier tasks
    sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                order = ['8-8-8 OPP', '8-8-8 Accuracy'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(outlier_file)
    plt.clf()

    # create plot for categorization tasks
    sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                order = ['AP', 'BLESS', 'Battig', 'ESSLI_2c', 'ESSLI_2b', 'ESSLI_1a'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(category_file)
    plt.clf()

    # create plot for analogy tasks
    sns.barplot(x = 'Test', y = 'Score',hue = 'Embedding', data = scores,
                order = ['MEN', 'WS353', 'WS353R', 'WS353S', 'SimLex999', 'RW', 'RG65', 'MTurk'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(similarity_file)

    plt.clf()

    # create plot for analogy tasks
    sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding',data = scores,
                order = ['Google', 'MSR', 'SemEval2012_2'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig(analogy_file)
    plt.clf()

# create graphs for a specific WEAT case for all models
def plot_specific_WEAT_comparison(file, test, out_file):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(file)
    labels = test

    # create plot
    plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                       order = labels,
                       hue_order = ['Original Finding', 'Original w2v', 'Debiased w2v', 'Original fastText', 'Debiased fastText',
                                    'Original GloVe', 'Debiased GloVe'],
                       palette = 'hls',
                       capsize = 0.05,
                       saturation = 8
                       )
    plt.ylim(0, 1.8)
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    plot.set_xticklabels(labels, rotation=45)
    plt.legend(loc='upper center')
    plt.setp(plot.get_legend().get_texts(), fontsize='8')
    plt.tight_layout()
    plt.savefig(out_file)

    plt.clf()

# create graph for all benchmark WEAT cases for a specific model
def plot_WEAT_benchmark_comparisons(input, model, graph_output):
    sns.set_context('paper')

    # load dataset
    all_scores = pd.read_csv(input)

    #filter by model
    original = all_scores[all_scores.Model == 'Original Finding']
    scores = all_scores[all_scores.Model == model]
    scores = pd.concat([original, scores])

    # create plot
    plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                       palette = 'hls',
                       capsize = 0.05,
                       saturation = 8,
                       errcolor = 'gray', errwidth = 2,
                       ci = 'sd'
                       )
    labels = ['Flowers/Insects & Pleasant/ Unpleasant', 'Instruments/ Weapons & Pleasant/ Unpleasant',
              'European Names/African Names & Pleasant/ Unpleasant', 'Male/Female & Career/Family',
              'Math/Arts & Male/Female', 'Science/Arts & Male/Female',
              'Mental Disease/ Physical Disease & Temporary/ Permanent',
              'Young Names/Old Names & Pleasant/ Unpleasant']
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    plot.set_xticklabels(labels,
                         rotation=90)
    plot.tick_params(axis='x', which='major', pad=10)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(graph_output)

    plt.clf()

# create graph for all political WEAT cases by categories for a specific model
def plot_WEAT_political_comparisons(input, model, output_graph1, output_graph2):
    sns.set_context('paper')

    # load dataset
    all_scores = pd.read_csv(input)

    #filter by model
    scores = all_scores[all_scores.Model == model]

    # create plot for generic cases
    labels = ['Unpleasant vs. Pleasant', 'Lazy vs. Hard Working', 'Evil vs. Good',
              'Family vs. Career', 'Female names vs. Male names', 'Female terms vs. Male terms',
              'STEM vs. Arts', 'Rich vs. Poor', 'Old people’s names vs. Young people’s names']
    generic_plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                               order = labels,
                               palette = 'hls',
                               capsize = 0.05,
                               saturation = 8,
                               errcolor = 'gray', errwidth = 2,
                               ci = 'sd'
                               )
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    generic_plot.set_xticklabels(labels,
                                 rotation=90)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(output_graph1)

    plt.clf()

    # create plot for specific cases
    labels = ['Liberal vs. Conservative', 'Climate Change Activist vs. Climate Change Denier',
              'Prochoice vs. Prolife', 'Peace vs. War', 'Pro-Immigration vs. Anti-Immigration',
              'Trust vs. Deceptive', 'Progressive vs. Moderate', 'Transparent vs. Secretive',
              'Atheist vs. Evangelical', 'Gun Control vs. Gun Rights',
              'Northern States vs. Southern States', 'Fair vs. Unfair',
              'Trickle Up Economics vs. Trickle Down Economics']
    specific_plot = sns.barplot(x = 'Test', y = 'Score', hue = 'Embedding', data = scores,
                                order = labels,
                                palette = 'hls',
                                capsize = 0.05,
                                saturation = 8,
                                errcolor = 'gray', errwidth = 2,
                                ci = 'sd'
                                )
    labels = [ '\n'.join(wrap(l, 16)) for l in labels ]
    specific_plot.set_xticklabels(labels,
                                  rotation=90)
    plt.ylabel('d Score')
    plt.tight_layout()
    plt.savefig(output_graph2)

    plt.clf()


def main():
    # sample for plotting and calculating WEAT becnhmark scores from a csv file
    plot_WEAT_benchmark_scores('weat_results/debiased/AllDebiasedWEATBenchmarkScores.csv',
                               'weat_results/debiased/debiasedWEAT_benchmark_plot',
                               'weat_results/debiased/mean_debiasedWEAT_benchmark.csv',
                               'weat_results/debiased/std_debiasedWEAT_benchmark.csv',
                               'weat_results/debiased/pvalue_debiasedWEAT_benchmark.csv')

    # sample for plotting and calculating WEAT political scores from a csv file
    plot_WEAT_political_scores('weat_results/debiased/AllDebiasedWEATPoliticalScores.csv',
                               'weat_results/debiased/debiasedWEAT_political_plot_generic',
                               'weat_results/debiased/debiasedWEAT_political_plot_specific',
                               'weat_results/debiased/mean_debiasedWEAT_political_scores.csv',
                               'weat_results/debiased/std_debiasedWEAT_political_scores.csv',
                               'weat_results/debiased/mean_debiasedWEAT_political_pvalues.csv',
                               'weat_results/debiased/std_debiasedWEAT_political_pvalues.csv')

    # sample for plotting and calculating eval scores from a csv file
    plot_eval_scores('eval_results/AllEvalDebiasedScores.csv',
                     'eval_results/debiased/debiased_outlier_graph',
                     'eval_results/debiased/debiased_category_graph',
                     'eval_results/debiased/debiased_similarity_graph',
                     'eval_results/debiased/debiased_analogy_graph',
                     'eval_results/debiased/debiased_mean_scores.csv',
                     'eval_results/debiased/debiased_std_scores.csv')

    #sample for plotting and calculating WEAT becnhmark scores from a csv file
    plot_specific_eval_comparison('../eval_results/AllMeanEvalScores.csv',
                                  'ESSLI_1a',
                                  'eval_results/comparisons/ESSLI_1a_graph')

    #sample for plotting WEAT gender cases from a csv file
    plot_specific_WEAT_comparison('weat_results/AllValidMeanBenchmarkWEATScores.csv',
                              ['Male/Female & Career/Family',
                               'Math/Arts & Male/Female', 'Science/Arts & Male/Female',],
                              'weat_results/comparisons/male_female_graph.png')

    #sample for plotting eval scores for w2v from a csv file
    plot_eval_comparison('eval_results/AllMeanEvalScores.csv',
                     'Original w2v',
                     'Debiased w2v',
                     'eval_results/comparisons/w2v_comparison_outlier_graph',
                     'eval_results/comparisons/w2v_comparison_category_graph',
                     'eval_results/comparisons/w2v_comparison_similarity_graph',
                     'eval_results/comparisons/w2v_comparison_analogy_graph')

    #sample for plotting WEAT becnhmark scores for w2v from a csv file
    plot_WEAT_benchmark_comparisons('weat_results/AllMeanBenchmarkWEATScores.csv',
                               'w2v',
                               'weat_results/comparisons/w2v_comparison_WEAT_benchmark_graph')

    #sample for plotting WEAT political scores for w2v from a csv file
    plot_WEAT_political_comparisons('weat_results/AllMeanPoliticalWEATScores.csv',
                                'w2v',
                                'weat_results/comparisons/w2v_comparison_WEAT_political_plot_generic',
                                'weat_results/comparisons/w2v_comparison_WEAT_political_plot_specific')



if __name__ == "__main__":
    main()