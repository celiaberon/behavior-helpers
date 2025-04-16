import configparser
import os
import platform
from pathlib import Path, PosixPath, WindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_config_variables(path_to_file: str,
                          section: str = 'color_palette') -> dict:

    '''
    Create dictionary containing parameter values that will be repeated
    across notebooks

    Args:
        section:
           Section name within configuration file.

    Returns:
        config_variables:
            Dictionary containing variables and assigned values from config
            file.
    '''

    # For color palette configuration only
    import matplotlib as mpl
    cpal = mpl.cm.RdBu_r(np.linspace(0, 1, 8))
    if not os.path.isfile(os.path.join(path_to_file, 'plot_config.ini')):
        path_to_file = os.getcwd()

    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(path_to_file, 'plot_config.ini'))
    # Create dictionary with key:value for each config item
    config_variables = {}
    for key in config_file[section]:
        config_variables[key] = eval(config_file[section][key])

    return config_variables


def downcast_all_numeric(df):

    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df


def cast_object_to_category(df):

    cols = df.select_dtypes('object').columns
    for col in cols:
        df[col] = df[col].astype('category')

    return df


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

    mapping = {(-1, 0): 'r', (-1, 1): 'R', (1, 0): 'l', (1, 1): 'L'}

    return ''.join([mapping[(c, r)] for c, r in zip(choices, rewards)])


def encode_sequences(rlseq, lag=3):

    mappings_ref_L = {'L': 'A', 'l': 'a', 'R': 'B', 'r': 'b'}
    mappings_ref_R = {'R': 'A', 'r': 'a', 'L': 'B', 'l': 'b'}

    if isinstance(rlseq, str):
        rlseq = list(rlseq)
    seqs = []
    seqs.extend([np.nan] * (lag - 1))
    for seq in np.array([rlseq[i:len(rlseq) - lag+i+1] for i in range(lag)]).T:
        ref = mappings_ref_L if seq[0].upper() == 'L' else mappings_ref_R
        seqs.append(''.join([ref[el] for el in seq]))

    return seqs


def reference_trials(ts_trials, trials, merge_var):

    '''
    Accurate referencing between separately loaded session-aggregated trials.
    '''

    ts_trials_ = ts_trials.copy()
    if 'Session' not in ts_trials_:
        ts_trials_ = ts_trials_.rename(columns={'session': 'Session'})

    ts_trials_ = ts_trials_.merge(trials[['Session', 'nTrial_orig', merge_var]],
                                  how='left', on=['Session', 'nTrial_orig'])
    return ts_trials_


def check_leg_duplicates(ax, coords=(0.8, 1)):

    h, lab = ax.get_legend_handles_labels()
    legend_reduced = dict(zip(lab, h))
    ax.legend(legend_reduced.values(), legend_reduced.keys(),
              bbox_to_anchor=coords, edgecolor='white')
    # plt.tight_layout()

def adjust_legend_scale(ax, patch_width=6, patch_height=4, coords=(1.0, 0), convert_labels='bool', fontsize=9, **kwargs):
    import matplotlib as mpl
    leg = ax.get_legend()
    handles = leg.legendHandles
    labels = [text.get_text() for text in leg.get_texts()]

    for handle in handles:
        if isinstance(handle, mpl.patches.Patch):
            handle.set_width(patch_width)
            handle.set_height(patch_height)
        elif isinstance(handle, mpl.lines.Line2D):
            kwargs['handlelength'] = patch_width

    if convert_labels=='bool':
        if all(label in ['True', 'False'] for label in labels):
            pass
        else:
            try:
                labels = [bool(float(label)) for label in labels if label not in ['True', 'False']]
            except ValueError:
                labels = [bool(label) for label in labels if label not in ['True', 'False']]

    elif convert_labels=='int':
        labels = [int(label) for label in labels]
    elif isinstance(convert_labels, dict):
        labels = [convert_labels[label] for label in labels]

    ax.legend(
        handles=handles, 
        labels=labels,
        bbox_to_anchor=coords,
        fontsize=fontsize,
        labelspacing=0.2,  # Vertical spacing
        handletextpad=0.4, # Horizontal spacing
        **kwargs)

    return ax

def convert_path_by_os(func):
    """Decorator to convert paths to appropriate format based on OS"""
    def wrapper(self, *args, **kwargs):
        path = func(self, *args, **kwargs)
        if isinstance(path, (str, Path, WindowsPath, PosixPath)):
            # Convert to string first to handle any path type
            path_str = str(path)
            if platform.system() == 'Windows':
                # Convert forward slashes to backslashes for Windows
                path_str = path_str.replace('/', '\\')
                if path_str.startswith('\\Volumes\\Neurobio\\MICROSCOPE\\'):
                    path_str = path_str.replace('\\Volumes\\Neurobio\\MICROSCOPE\\', 'N:\\MICROSCOPE\\')
                return WindowsPath(path_str)
            else:
                # For non-Windows systems, ensure forward slashes
                path_str = path_str.replace('\\', '/')
                return PosixPath(path_str)
        return path
    return wrapper


def convert_path_by_os_direct(path):
    if isinstance(path, (str, Path, WindowsPath, PosixPath)):
        path_str = str(path)
        if platform.system() == 'Windows':
            path_str = path_str.replace('/', '\\')
            if path_str.startswith('\\Volumes\\Neurobio\\MICROSCOPE\\'):
                path_str = path_str.replace('\\Volumes\\Neurobio\\MICROSCOPE\\', 'N:\\MICROSCOPE\\')
            return WindowsPath(path_str)
        else:
            path_str = path_str.replace('\\', '/')
            if path_str.startswith('N:/MICROSCOPE/'):
                path_str = path_str.replace('N:/MICROSCOPE/', '/Volumes/Neurobio/MICROSCOPE/')
            return PosixPath(path_str)
    return path