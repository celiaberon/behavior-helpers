import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_lick_raster(df: pd.DataFrame,
                     fs: int | float,
                     state: str = 'Cue',
                     session_id: str = None,
                     start_trial: int = 0,
                     n_trials: int = 120):

    pal = sns.color_palette('Greys')
    colors = {1: pal[3], 2: 'k', 3: 'g'}  # color codes for choice direction

    inc_height = 1.  # displacement on y axis for each trial
    window = (-1, 2)
    fs = int(fs)

    if not isinstance(session_id, str):
        session_id = np.random.choice(df.Session.unique())
        print(session_id)
    df_sess = df.copy().query('Session == @session_id')

    # Align all trials to onset of state, and take only n_trials of interest.
    aligned_idx = (df_sess.query(f'{state} == 1')
                   .groupby('nTrial', as_index=False)
                   .nth(0).index.astype('int'))
    end_trial = min((start_trial + n_trials, df_sess.nTrial.values[-1]))
    n_trials = end_trial - start_trial
    aligned_idx = aligned_idx[start_trial:end_trial].copy()

    fig, ax = plt.subplots(figsize=(4, len(aligned_idx) / 45),
                           layout='constrained')

    n = 0  # trial counter
    for idx in aligned_idx[1:-1]:
        licks = df_sess.loc[idx + (window[0]*fs): idx + (window[1]*fs) - 1].copy()
        licks['aligned_ms'] = np.arange(*window, step=1 / fs)
        licks = licks.loc[licks.iSpout != 0].copy()

        ax.scatter(licks['aligned_ms'],  # x-axis plots in ms where licks occur
                   np.ones(len(licks)) + n,  # plots trial at same y-position
                   c=[colors[sp] for sp in licks.iSpout],  # color by direction
                   marker='|', alpha=0.9, s=4, label='_nolegend_')
        n += inc_height

    ax.text(window[1]+0.15, n_trials-10, 'lick left', color=pal[3])
    ax.text(window[1]+0.15, n_trials-30, 'lick right', color='k')
    if state == 'Cue':
        cue_duration = 0.075  # in seconds
        ax.text(window[0]+0.5, n_trials+10, 'ENL', ha='center')
        ax.text(1, n_trials+10, 'Consumption', ha='center', color='k')
        ax.fill_betweenx(y=[0, n+4], x1=0, x2=0+cue_duration, alpha=0.4,
                         color=sns.color_palette('colorblind')[3],
                         edgecolor=None)

    ax.text(0, n_trials + 10, state, ha='center',
            color=sns.color_palette('colorblind')[3] if state == 'Cue' else 'k')
    ax.set(ylabel='Trial', xlabel='time (s)', xlim=window,
           ylim=(0, n+5), xticks=[-1, 0, 1, 2])
    sns.despine()

    return fig, ax, aligned_idx
