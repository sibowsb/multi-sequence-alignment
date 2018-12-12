import pandas as pd
from time import time
from random import random, choice
from greedy_aligner import *
import SeqAlign     # from https://github.com/niemasd/SeqAlign


def gen_test_input(num_seqs, max_length, diff_rate, alphabet='ATCG'):
    """
    Generate a randomized test input.

    Parameters
    ----------
    num_seqs : int
        Number of sequences to generate.
    max_length : int
        Maximum length of sequence.
    diff_rate : float
        The fractional rate at which a mismatch occurs.
    alphabet : str, optional
        The alphabet of allowed characters. The default is ``ATCG``.

    Returns
    -------
    dict
        A dictionary of (label: sequence) pairs.

    """
    true_seq = [choice(alphabet) for i in range(max_length)]
    seqs = {}
    for i in range(num_seqs):
        curr_seq = []
        for j in range(max_length):
            if random() < diff_rate:
                curr_seq.append(choice([x for x in alphabet] + ['']))
            else:
                curr_seq.append(true_seq[j])
            seqs['v%d' % i] = ''.join(curr_seq)
    return seqs


def timed_run(num_seqs, max_length, diff_rate):
    """
    Execute alignment with timing.
    
    Returns
    -------
    float
        Amount of time, in seconds, it took my implementation to run.
    float
        Amount of time, in seconds, it took SeqAlign to run.
    """
    input_seqs = gen_test_input(num_seqs, max_length, diff_rate)
    
    # Running my implementation
    strs = list(input_seqs.values())
    start = time()
    p = multi_align(strs)
    my_time = time() - start

    # Running SeqAlign
    start = time()
    SeqAlign.multiAlign(input_seqs, gap=-1, treefile=None)
    seqalign_time = time() - start

    return my_time, seqalign_time


def run_comparison():
    num_seqs = [2, 4, 8, 12]
    max_lengths = [4, 8, 16, 32, 64, 128, 256]
    records = []
    for n in num_seqs:
        for l in max_lengths:
            print('Running %d sequences of length %d' % (n, l))
            my_time, its_time = timed_run(n, l, 0.1)
            records.append([n, l, my_time, its_time])
            print('  mine finished in %.2f secs; theirs finished in %.2f secs'
                        % (my_time, its_time))
    df = pd.DataFrame()
    df.index.name = 'run'
    df['num_seqs'] = [x[0] for x in records]
    df['max_len'] = [x[1] for x in records]
    df['my_time'] = [x[2] for x in records]
    df['its_time'] = [x[3] for x in records]
    df.to_csv('data/benchmark_comparison_results.csv')


def viz():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = pd.read_csv('data/benchmark_comparison_results.csv')
    num_seqs = [2, 4, 8, 12]
    max_len = [4, 8, 16, 32, 64, 128, 256]
    markers = ['v', 'o', 's', '*']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for n, marker in zip(num_seqs, markers):
        sel = data[data['num_seqs'] == n]
        plt.plot(sel['max_len'], sel['my_time'], color='blue',
                 marker=marker, linestyle=':')
        plt.plot(sel['max_len'], sel['its_time'], color='red',
                 marker=marker, linestyle=':',)
    plt.xlabel('Sequence Length')
    plt.ylabel('Running Time [sec]')
    plt.title('Linear Scale')

    plt.subplot(1, 2, 2)
    for n, marker in zip(num_seqs, markers):
        sel = data[data['num_seqs'] == n]
        plt.loglog(sel['max_len'], sel['my_time'], color='blue',
                   marker=marker, linestyle=':')
        plt.loglog(sel['max_len'], sel['its_time'], color='red',
                   marker=marker, linestyle=':')
    plt.xlabel('Sequence Length')
    plt.ylabel('Running Time [sec]')
    plt.title('Log Scale')

    items = [
        Line2D([0], [0], color='red', label='SeqAlign'),
        Line2D([0], [0], color='blue', label='Mine'),
        Line2D([0], [0], color='gray', linestyle=':', marker='v', label='2 sequences'),
        Line2D([0], [0], color='gray', linestyle=':', marker='o', label='4 sequences'),
        Line2D([0], [0], color='gray', linestyle=':', marker='s', label='8 sequences'),
        Line2D([0], [0], color='gray', linestyle=':', marker='*', label='12 sequences'),
    ]
    plt.legend(handles=items, loc='center right', bbox_to_anchor=(1.4, 0.5))

    plt.tight_layout()
    plt.savefig('data/scaling_plot.svg')


if __name__ == '__main__':
    run_comparison()
    viz()

