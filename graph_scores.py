import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

def plot_eval_scores(file):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(file)

    # create plot
    sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = ['8-8-8 OPP', '8-8-8 Accuracy '],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig('outlier_graph')

    plt.clf()

    sns.barplot(x = 'Test', y = 'Score', hue = 'Model', data = scores,
                order = ['AP', 'BLESS', 'Battig', 'ESSLI_2c', 'ESSLI_2b', 'ESSLI_1a'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig('category_graph')
    plt.clf()
    sns.barplot(x = 'Test', y = 'Score',hue = 'Model', data = scores,
                order = ['MEN', 'WS353', 'WS353R', 'WS353S', 'SimLex999', 'RW', 'RG65', 'MTurk'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig('similarity_graph')
    plt.clf()
    sns.barplot(x = 'Test', y = 'Score', hue = 'Model',data = scores,
                order = ['Google', 'MSR', 'SemEval2012_2'],
                palette = 'hls',
                capsize = 0.05,
                saturation = 8,
                errcolor = 'gray', errwidth = 2,
                ci = 'sd'
                )
    plt.savefig('analogy_graph')

    print(scores.groupby(['Test', 'Model']).mean()['Score'])
    print(scores.groupby(['Test', 'Model']).std()['Score'])


def plot_WEAT_benchmark_scores(input, output):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(input)

    # create plot

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
    plt.savefig(output)

    plt.clf()

    scores.groupby(['Test', 'Model']).mean()['Score'].to_csv('mean_WEAT_benchmark_scores.csv')
    scores.groupby(['Test', 'Model']).std()['Score'].to_csv('std_WEAT_benchmark_scores.csv')

def plot_WEAT_political_scores(input, output_graph1, output_graph2, mean_output, std_output, pvalue_mean_output,
                               pvalue_std_output):
    sns.set_context('paper')

    # load dataset
    scores = pd.read_csv(input)

    # create plot
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

    # create plot
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

    scores.groupby(['Test', 'Model']).mean()['Score'].to_csv(mean_output)
    scores.groupby(['Test', 'Model']).std()['Score'].to_csv(std_output)
    scores.groupby(['Test', 'Model']).mean()['pValue'].to_csv(pvalue_mean_output)
    scores.groupby(['Test', 'Model']).std()['pValue'].to_csv(pvalue_std_output)


def main():
    #plot_WEAT_benchmark_scores('AllWEATBenchmarkScores.csv', 'WEAT_benchmark_plot')
    plot_WEAT_political_scores('weat_results/original/AllWEATPoliticalScores.csv',
                               'weat_results/original/WEAT_political_plot_generic',
                               'weat_results/original/WEAT_political_plot_specific',
                               'weat_results/original/mean_WEAT_political_scores.csv',
                               'weat_results/original/std_WEAT_political_scores.csv',
                               'weat_results/original/mean_WEAT_political_pvalues.csv',
                               'weat_results/original/std_WEAT_political_pvalues.csv')

if __name__ == "__main__":
    main()