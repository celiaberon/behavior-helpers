import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_lick_raster(df, fs, state='Cue', start_trial=0, n_trials=120):

    from matplotlib.lines import Line2D
    pal = sns.color_palette('Greys')
    colors = {1: pal[3], 2: 'k', 3: 'g'} # color codes for choice direction

    inc_height = 1.  # displacement on y axis for each trial
    window = (-1, 2)

    # Align all trials to onset of state, and take only n_trials of interest.
    aligned_idx = (df.query(f'{state} == 1')
                   .groupby('nTrial', as_index=False)
                   .nth(0).index.astype('int'))
    aligned_idx = aligned_idx[start_trial:start_trial + n_trials].copy()

    fig, ax = plt.subplots(figsize=(4, len(aligned_idx) / 45), layout='constrained')

    n = 0  # trial counter
    for idx in aligned_idx[1:-1]:
        licks = df.loc[idx + (window[0] * int(fs)): idx + (window[1] * int(fs)) - 1].copy()
        licks.index = pd.Index(np.arange(window[0], (window[1]), step=1 / int(fs)))
        licks = licks.loc[licks.iSpout != 0].copy()

        ax.scatter(licks.index,  # x-axis plots in ms where licks occur
                   np.ones(len(licks)) + n,  # plots all at same position on y-axis for single trial
                   c=[colors[sp] for sp in licks.iSpout],  # color codes by direction
                   marker='|', alpha=0.9, s=4, label='_nolegend_')
        n += inc_height

    ax.text(window[1] + 0.15, n_trials - 10, 'lick left', color=pal[3])
    ax.text(window[1] + 0.15, n_trials - 30, 'lick right', color='k')
    if state == 'Cue':
        cue_duration = 0.075 # in seconds
        ax.text(window[0] + 0.5, n_trials + 10, 'ENL', ha='center')
        ax.text(1, n_trials + 10, 'Consumption', ha='center', color='k')
        ax.fill_betweenx(y=[0, n + 4], x1=0, x2=0 + cue_duration, alpha=0.4,
                     color=sns.color_palette('colorblind')[3],
                     edgecolor=None)

    ax.text(0, n_trials + 10, state, ha='center',
            color=sns.color_palette('colorblind')[3] if state=='Cue' else 'k')
    ax.set(ylabel='Trial', xlabel='time (s)', xlim=window,
           ylim=(0, n + 5), xticks=[-1, 0, 1, 2])
    sns.despine()

    return fig, ax, aligned_idx


def label_bout_group(timeseries, trials, bout_ili_threshold=0.4):

    '''Take all licks with ILI over threshold and define them as onset of new 
    bout, which each get own numberic ID. Define consumption bout as first 
    bout in consumption period.
    '''
    timeseries_ = timeseries.copy()
    trials_ = trials.copy()
    timed_licks = timeseries.dropna(subset='ILI').copy()
    timed_licks['lick_bout_thresh'] = timed_licks['ILI'] > bout_ili_threshold
    timed_licks['bout_group'] = timed_licks.lick_bout_thresh.cumsum()
    consumption_bouts = (timed_licks.query('stateConsumption==1 & ~iSpout.isna()')
                                    .groupby('nTrial')
                                    .nth(0)['bout_group'])

    timeseries_.loc[timed_licks.index.values, 'bout_group'] = timed_licks['bout_group']
    timeseries_['consumption_bout'] = timeseries_['nTrial'].map(consumption_bouts)
    consumption_nLicks = (timeseries_.query('bout_group==consumption_bout')
                                     .groupby('nTrial')['iSpout']
                                     .count())
    trials_['cons_nLicks'] = trials_['nTrial'].map(consumption_nLicks)

    return timeseries_, trials_