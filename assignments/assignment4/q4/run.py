import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import e


def hash_func(a, b, n_buckets, x, p=123457):
    y = x % p
    hash_val = (a*y + b) % p
    return hash_val % n_buckets


def hash_item(i_id, h_func, h_func_params, n_buckets, counts_dict):
    for key in h_func_params.keys():
        a, b = h_func_params[key]
        i_id_hash = h_func(a, b, n_buckets, i_id)
        counts_dict[(key, i_id_hash)] += 1


def approximate_f_i(i_id, h_func, h_func_params, n_buckets, counts_dict):
    min_val = np.inf
    for key in h_func_params.keys():
        a, b = h_func_params[key]
        i_id_hash = h_func(a, b, n_buckets, i_id)
        if counts_dict[(key, i_id_hash)] < min_val:
            min_val = counts_dict[(key, i_id_hash)]
    return min_val


def plot_log_error_vs_log_item_freq(error_list, item_freq_list):
    plt.figure(figsize=(20, 10))
    plt.loglog(item_freq_list, error_list, "2")
    plt.title('Log Relative Error vs Log Item Frequency')
    plt.xlabel('Log Item Frequency')
    plt.ylabel('Log Relative Error')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    # Load data
    path = 'data/'
    hash_params_dict = {idx: (int(x[0]), int(x[1])) for idx, x in enumerate(np.loadtxt(path + 'hash_params.txt').tolist())}

    # Init
    delta = np.exp(-5)
    eps = e * 1e-4
    num_hash_funcs = np.log(1/delta).astype(int)
    num_buckets = np.ceil(e / eps).astype(int)
    hash_counts_dict = {x: 0 for x in itertools.product(np.arange(num_hash_funcs), np.arange(num_buckets))}

    # Hash stream data
    t = 0
    with open(path + 'words_stream.txt', 'r', encoding='utf8') as stream:
        for row in stream:
            t += 1
            item_id = int(row)
            hash_item(item_id, hash_func, hash_params_dict, num_buckets, hash_counts_dict)

    # Approximate f_i and get relative error e_i. Get item freqs as well.
    e_i = []
    item_freq = []
    with open(path + 'counts.txt', 'r', encoding='utf8') as counts_data:
        for row in counts_data:
            item_id, f_i = [int(x) for x in row.split('\t')]
            f_i_approx = approximate_f_i(item_id, hash_func, hash_params_dict, num_buckets, hash_counts_dict)
            e_i.append((f_i_approx - f_i) / f_i)
            item_freq.append(f_i / t)

    # Plot relate error vs item frequency
    plot_log_error_vs_log_item_freq(e_i, item_freq)
