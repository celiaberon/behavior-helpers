import numpy as np


def calc_sem(x):

    return np.std(x) / np.sqrt(len(x))


def calc_ci(x, confidence: float = 0.95):

    from scipy.stats import norm

    # determine critical value for set confidence interval.
    alpha_2 = (1 - confidence) / 2
    critical_value = norm.ppf(1 - alpha_2)

    sem = calc_sem(x)
    err_width = critical_value * sem
    ci_low = np.mean(x) - err_width
    ci_high = np.mean(x) + err_width
    return (ci_low, ci_high, err_width)


def make_onehot_array(x):

    if len(np.unique(x)) == 1:
        return x

    onehot = np.zeros((x.size, x.max() + 1))
    onehot[np.arange(x.size), x] = 1
    return onehot


def get_dict_item_by_idx(d, idx):

    for i, (k, v) in enumerate(d.items()):
        if i == idx:
            return k, v
        

def encode_as_rl(choices, rewards):
        
    mapping = {(-1,0): 'r', (-1,1): 'R', (1,0): 'l', (1,1): 'L'} 

    return ''.join([mapping[(c,r)] for c,r in zip(choices, rewards)])

def encode_sequences(rlseq, lag=3):

    mappings_ref_L = {'L': 'A', 'l': 'a', 'R': 'B', 'r': 'b'}
    mappings_ref_R = {'R': 'A', 'r': 'a', 'L': 'B', 'l': 'b'}

    seqs = []
    for first, second, third in zip(rlseq[2:], rlseq[1:-1], rlseq[:-2]):

        ref = mappings_ref_L if first.upper()=='L' else mappings_ref_R
        seqs.append(''.join([ref[first], ref[second], ref[third]]))

    return seqs
