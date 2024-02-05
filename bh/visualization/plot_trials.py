import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
        'axes.titlesize': 12, 'savefig.transparent': True})


def binomial_err(p, n):

    return np.sqrt(p * (1 - p)) / n


def calc_conditional_probs(trials: pd.DataFrame,
                           htrials: int = 1,
                           err_func: str = 'sem',
                           pred_col: str = '+1switch',
                           add_grps: str = None,
                           sortby: str = None,
                           **kwargs) -> pd.DataFrame:

    '''
    Calculate average probability of action (or more generally, event)
    conditioned on grouping column as:
         P(pred_col | grp_cols, add_grps)
    Grouping column currently fixed to sequential (or single) encoded
    action-outcome event.

    Args:
        trials:
            Trial-based dataframe.
        htrials:
            Number of trials in history to condition on. Defaults to 1, for
            most recent trial only.
        err_func:
            Error function to use on conditional prob. Supports binom for
            binomial error of conditional prob, or sem for standard error of
            conditioned seq distribution.
        pred_col:
            Variable for which the conditional probability is calculated.
        add_grps:
            Additional groups on which probability is conditioned.
        sortby:
            Whether to sort conditioned probabilities. Can support either by
            conditional prob column, or pre-sorted conditioned history
            backbone.
    Returns:
        cond_probs:
            Table of conditional probabilities and their conditioned group.
    '''

    grp_cols = [f'seq{htrials}']
    if add_grps:
        grp_cols.append(add_grps)

    # Calculate average probability of action (or predicted event) conditioned
    # on group.
    cond_probs = (trials.groupby(grp_cols)[pred_col]
                  .agg(['mean', 'std', 'count'], )
                  .reset_index()
                  .rename(columns={f'seq{htrials}': 'history',
                                   'mean': 'pevent',
                                   'count': 'n'})
                  )

    if err_func == 'binom':
        cond_probs['pevent_err'] = binomial_err(cond_probs.pevent.values,
                                                cond_probs.n.values)
    elif err_func == 'sem':
        cond_probs['pevent_err'] = cond_probs['std'] / np.sqrt(cond_probs['n'])
    else:
        raise NotImplementedError

    if sortby == 'pevent':
        cond_probs = cond_probs.sort_values(by='pevent')
    elif sortby == 'history':
        horder = kwargs.get('order', None)
        cond_probs.history = cond_probs.history.astype('category')
        cond_probs['history'] = cond_probs['history'].cat.set_categories(horder)
        cond_probs = cond_probs.sort_values(by='history')

    return cond_probs


def plot_sequences(cond_probs: pd.DataFrame,
                   overlay: pd.DataFrame = None,
                   **kwargs):

    '''
    Plot conditional probabilities.
    Args:
        cond_probs:
            Table of conditional probabilities.
        overlay:
            Additional table of conditional probabilities to overlay on plot.
    Returns:
        fig, ax:
            Matplotlib objects containing conditional probability plot.
    '''

    sns.set(style='ticks', font_scale=1.0,
            rc={'axes.labelsize': 12, 'axes.titlesize': 12},
            palette='deep')

    overlay_label = kwargs.get('overlay_label', '')
    yval = kwargs.get('yval', 'pevent')

    if cond_probs.history.nunique() < 10:
        fig, ax = plt.subplots(figsize=(4.2, 2.5))
    else:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    if overlay is not None:
        sns.barplot(x='history', y=yval, data=overlay, ax=ax,
                    color=sns.color_palette()[0],
                    label=overlay_label,
                    alpha=1.0)
        ax.errorbar(x='history', y=yval, data=overlay, yerr=f'{yval}_err',
                    fmt=' ', label=None, color=sns.color_palette('dark')[0])

    sns.barplot(x='history', y=yval, data=cond_probs, ax=ax, color='k',
                label=kwargs.get('main_label', ''),
                alpha=kwargs.get('alpha', 0.4), edgecolor=None)
    ax.errorbar(x='history', y=yval, data=cond_probs, yerr=f'{yval}_err',
                fmt=' ', label=None, color='k')

    ax.set(xlim=(-1, len(cond_probs)), ylim=(0, 1),
           ylabel=kwargs.get('ylab', 'P(switch)'),
           title=kwargs.get('title', None))
    plt.tight_layout()
    sns.despine()

    if overlay_label:
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    if kwargs.get('rotate_x', False):
        plt.xticks(rotation=90)

    return fig, ax


def plot_sequence_points(cond_probs,
                         yval: str = 'pevent',
                         fig=None,
                         ax=None,
                         **kwargs):

    if ax is None:
        if cond_probs.history.nunique() < 10:
            fig, ax = plt.subplots(figsize=(4.2, 2.5))
        else:
            fig, ax = plt.subplots(figsize=(10, 2.5))

    sns.swarmplot(data=cond_probs, x='history', y=yval, ax=ax,
                  hue=kwargs.get('grp', 'history'), palette='Blues', size=5)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    return fig, ax


def plot_single_session(trials: pd.DataFrame,
                        tstart: int = 0,
                        session_id: str = None,
                        ntrials: int = None):

    '''
    Plot action-outcome sequences across a session, overlaid with system state

    Args:
        trials:
            Trial table containing choice, reward, and state info.
        tstart:
            Optional to set trial ID on which to start the plot.
        session_id:
            Optional to set which session to plot. Default random.
        ntrials:
            Optional to set number trials to plot. Default to end of session.
    Returns:
        fig, ax:
            Matplotlib objects contianing session plot.

    '''
    shade_color = sns.color_palette('Greys', n_colors=6)[1]

    # Choose a random session if none provided and multipe session in df.
    if (session_id is None) and (trials.Session.dropna().nunique() > 1):
        session_id = np.random.choice(trials.Session.dropna().unique())
    session = trials.query('Session==@session_id').copy()

    if ntrials is None:
        ntrials = len(session)  # default is full session

    # use original trial numbers if plotting from agg data
    if 'nTrial_orig' in trials.columns:
        session = session.set_index('nTrial_orig')
    elif session.index.name != 'nTrial':
        session = session.set_index('nTrial')  # use nTrial as index

    # Trim session to designated trial bounds.
    tstart = max(session.index.min(), tstart)
    tstop = min(tstart + ntrials, session.index.max())
    session = session.loc[tstart: tstop].copy()

    # Locate first trial in every block.
    bidx = session.query('iInBlock == 0').index
    if bidx[0] > 1:
        bidx = np.insert(bidx, 0, 0)
    if bidx[-1] != len(session):
        bidx = np.insert(bidx, len(bidx), len(session))

    fig, ax = plt.subplots(figsize=[7.5, 1.3])
    # Iterate over blocks and shade odd blocks only (left blocks).
    for bstart, bstop in zip(bidx[:-1], bidx[1:]):
        if session.loc[bstop - 1, 'State'] % 2 == 0:
            continue
        ax.fill_betweenx([-1, 3.], bstart - 0.5, bstop - 0.5,
                         color=shade_color, label=None)

    session['reward_size'] = (session['Reward'] + 1) * 0.8
    sns.scatterplot(data=session, x=session.index.name, y='direction',
                    size='reward_size', ax=ax, color='k')
    sns.scatterplot(data=session.query('timeout==1'), x='nTrial_orig', y=0.5,
                    marker='x', ax=ax, color='k', label='timeout')

    ax.set(xlim=(tstart, tstop), xlabel='Trial', ylim=(-0.1, 1.1), ylabel='')
    plt.yticks([0.05, 0.95], ['right', 'left'], va='center')
    y = ax.get_yticklabels()
    y[1].set_backgroundcolor(shade_color)
    y[0].set_bbox(dict(facecolor='none', edgecolor='black'))
    ax.tick_params(axis='y', which='both', length=0)
    handles, labels = ax.get_legend_handles_labels()
    trial_type = {'0.8': 'unrewarded', '1.6': 'rewarded', 'timeout': 'timeout'}
    ax.legend(handles=handles, labels=[trial_type[label] for label in labels],
              bbox_to_anchor=(1, 1), loc='upper left', frameon=False,
              edgecolor=None, title='trial type')

    return fig, ax


def calc_bpos_probs(trials, add_agg_cols=None, add_cond_cols=None):

    agg_cols = ['Switch', 'selHigh']
    if add_agg_cols is not None:
        if not isinstance(add_agg_cols, list):
            add_agg_cols = [add_agg_cols]
        agg_cols.extend(add_agg_cols)
    agg_funcs = {agg_col: np.mean for agg_col in agg_cols}

    # Add column tracking how far trial is from end of block.
    trials['rev_iInBlock'] = trials['iInBlock'] - trials['blockLength']

    grp_forward = ['iInBlock']
    grp_rev = ['rev_iInBlock']
    if add_cond_cols is not None:
        if not isinstance(add_cond_cols, list):
            add_cond_cols = [add_cond_cols]
        grp_forward.extend(add_cond_cols)
        grp_rev.extend(add_cond_cols)

    bpos_probs = (trials
                  .groupby(grp_forward, as_index=False)
                  .agg(agg_funcs))
    bpos_probs_rev = (trials
                      .groupby(grp_rev, as_index=False)
                      .agg(agg_funcs))

    # Combine negative block positions with forward-counting positions.
    bpos_probs_rev = bpos_probs_rev.rename(columns={'rev_iInBlock': 'iInBlock'})
    bpos = pd.concat((bpos_probs_rev, bpos_probs)).sort_values(by='iInBlock')

    return bpos


def plot_bpos_behavior(bpos_probs,
                       include_units: str = '',
                       plot_features: dict = None):

    if plot_features is None:
        plot_features = {'selHigh': ('P(high port)', (0, 1)),
                         'Switch': ('P(switch)', (0, 0.4))}

    n_plots = len(plot_features.keys())
    fig, axs = plt.subplots(ncols=n_plots, figsize=(3.6 * n_plots, 2.4))

    for i, (metric, ax_vars) in enumerate(plot_features.items()):
        if include_units:
            sns.lineplot(bpos_probs, x='iInBlock', y=metric, ax=axs[i],
                         color='k', units=include_units, estimator=None,
                         lw=0.6, label=include_units)
        sns.lineplot(bpos_probs, x='iInBlock', y=metric, ax=axs[i], color='k',
                     lw=2, label='pooled')

        label, ylim = ax_vars
        axs[i].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        axs[i].set(xlabel='block position', xlim=(-10, 10),
                   ylabel=label, ylim=ylim)

    sns.despine()
    axs[0].legend().remove()
    axs[-1].legend(bbox_to_anchor=(1, 1), edgecolor='white')
    check_leg_duplicates(axs[-1])


    plt.tight_layout()

    return fig, axs


def check_leg_duplicates(ax):

    h, l = ax.get_legend_handles_labels()
    legend_reduced = dict(zip(l, h))
    ax.legend(legend_reduced.values(), legend_reduced.keys(),
              bbox_to_anchor=(0.8, 1), edgecolor='white')
    plt.tight_layout()
