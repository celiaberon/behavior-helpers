import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bh.utils import check_leg_duplicates

sns.set_theme(style='ticks',
    font_scale=1.0,
    rc={'axes.labelsize': 10,
        'axes.titlesize': 10,
        'savefig.transparent': True,
        'legend.title_fontsize': 10,
        'legend.fontsize': 10,
        'legend.borderpad': 0.2,
        'legend.frameon': False,
        'figure.titlesize': 10,
        'figure.subplot.wspace': 0.1,
        })

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
            most recent trial only. Alternatively, column name directly.
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

    if isinstance(htrials, int):
        grp_cols = [f'seq{htrials}']
    else:
        grp_cols = [htrials]
    if add_grps:
        grp_cols.append(add_grps)

    # Calculate average probability of action (or predicted event) conditioned
    # on group.
    cond_probs = (trials.groupby(grp_cols, observed=True)[pred_col]
                  .agg(['mean', 'std', 'count'], )
                  .reset_index()
                  .rename(columns={grp_cols[0]: 'history',
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

    cond_probs['history'] = cond_probs['history'].astype('str')  # for sorting
    cols = [col for col in cond_probs.columns if cond_probs[col].dtype != 'category']
    cond_probs[cols] = cond_probs[cols].fillna(0)

    if sortby == 'pevent':
        cond_probs = cond_probs.sort_values(by='pevent')
    elif sortby == 'history':
        # If explicit order for histories already provided.
        horder = kwargs.get('order', None)
        if add_grps:
            other_cols = [col for col in cond_probs.columns
                          if col not in ['history', add_grps]]
            cp_all_histories = pd.DataFrame()
            # Make sure each group has same histories, even if some types
            # missing.
            for g, cp_grp in cond_probs.groupby(add_grps, observed=True):
                if missing_h := set(horder) - set(cp_grp.history):
                    cp_grp = (pd.concat(
                        (cp_grp, pd.DataFrame({'history': list(missing_h)},
                         index=np.arange(len(missing_h)))))
                        .reset_index(drop=True))
                    cp_grp.loc[cp_grp.history.isin(list(missing_h)), other_cols] = 0
                    cp_grp.loc[cp_grp.history.isin(list(missing_h)), add_grps] = g
                cp_all_histories = (pd.concat((cp_all_histories, cp_grp))
                                    .reset_index(drop=True))
            cond_probs = cp_all_histories.copy()
            grpby = ['history', add_grps]
        else:
            grpby = 'history'
        cond_probs.history = cond_probs.history.astype('category')
        cond_probs['history'] = cond_probs['history'].cat.set_categories(horder)
        cond_probs = cond_probs.sort_values(by=grpby)

    return cond_probs


def plot_sequences(cond_probs: pd.DataFrame,
                   overlay: pd.DataFrame = None,
                   yval: str = 'pevent',
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
            rc={'axes.labelsize': 11, 'axes.titlesize': 11},
            palette='deep')

    overlay_label = kwargs.get('overlay_label', '')
    hue = kwargs.get('hue', [])
    if kwargs.get('ax', None) is None:
        # fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
        if cond_probs.history.nunique() < 10:
            fig, ax = plt.subplots(figsize=(4.2, 2.5), layout='constrained')
        else:
            fig, ax = plt.subplots(figsize=(10, 2.5), layout='constrained')
    else:
        ax = kwargs.pop('ax')
        fig = None

    if hue:
        err_col = 'err_pos'  # column to contain x position values
        nbars = cond_probs['history'].nunique()
        err_pos_low = np.arange(nbars) - 0.2
        err_pos_high = np.arange(nbars) + 0.2
        err_pos = list(itertools.chain(*[(low, high) for low, high
                                         in zip(err_pos_low, err_pos_high)]))
        cond_probs[err_col] = err_pos
    else:
        err_col = 'history'

    if overlay is not None:
        sns.barplot(x='history', y=yval, data=overlay, ax=ax,
                    color=sns.color_palette()[0],
                    label=overlay_label,
                    alpha=1.0)
        ax.errorbar(x='history', y=yval, data=overlay, yerr=f'{yval}_err',
                    fmt=' ', label=None, color=sns.color_palette('dark')[0])

    sns.barplot(x='history', y=yval, data=cond_probs, ax=ax, color='k',
                label=kwargs.get('main_label', ''), legend=1 - any(hue),
                hue=hue if hue else None,
                palette=kwargs.get('palette', None),
                alpha=kwargs.get('alpha', 0.4), edgecolor=None)
    ax.errorbar(x=err_col, y=yval, data=cond_probs, yerr=f'{yval}_err',
                fmt=' ', label=None, color='k')

    ax.set(xlim=(-1, len(cond_probs) // (any(hue) + 1)), ylim=(0, 1),
           ylabel=kwargs.get('ylab', 'P(switch)'),
           title=kwargs.get('title', None))
    sns.despine()

    if overlay_label:
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    if kwargs.get('rotate_x', False):
        plt.xticks(rotation=90)

    return fig, ax


def plot_sequence_points(cond_probs: pd.DataFrame,
                         yval: str = 'pevent',
                         fig=None,
                         ax=None,
                         grp: str = 'Mouse',
                         **kwargs):

    '''
    Swarmplot representation of conditional probabilities.
    Args:
        cond_probs:
            Table of conditional probabilities.
        yval:
            Column containing dependent variable.
        fig:
            Matplotlib figure object if overlaying on existing figure.
        ax:
            Matplotlib axis object if overlaying on existing axis.
    Returns:
        fig, ax:
            Matplotlib figure objects containing plot.
    '''

    if ax is None:
        if cond_probs.history.nunique() < 10:
            fig, ax = plt.subplots(figsize=(4.2, 2.5))
        else:
            fig, ax = plt.subplots(figsize=(10, 2.5))

    if 'size' not in kwargs:
        # Point size as function of number of points.
        kwargs['size'] = np.min((5, 20 / cond_probs[grp].nunique()))

    if cond_probs[grp].nunique() <= 10:
        sns.swarmplot(data=cond_probs.sort_values(by=grp), x='history', y=yval,
                      ax=ax, hue=grp, **kwargs)
    else:
        sns.stripplot(data=cond_probs.sort_values(by=grp), x='history', y=yval,
                      ax=ax, hue=grp, **kwargs)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    ax.set(ylim=(0, 1))
    sns.despine()

    return fig, ax


def plot_seq_bar_and_points(trials: pd.DataFrame,
                            htrials: int = 1,
                            pred_col: str = '+1switch',
                            grp: str = 'Mouse',
                            ax=None,
                            **kwargs):

    '''
    Handles pairing of common base conditional prob barplot with sub-group
    pointplots.
    '''

    # Calculate aggregate policies and plot conditional probs as barplot.
    x_histories = kwargs.pop('x_histories', None)
    pooled_policies = calc_conditional_probs(trials,
                                             htrials,
                                             pred_col=pred_col,
                                             sortby='history' if np.any(x_histories)
                                                    else 'pevent',
                                             order=x_histories)
    x_histories = pooled_policies.history.values
    fig, ax = plot_sequences(pooled_policies, alpha=0.5, title='', ax=ax)

    # Calculate subgroup policies that will be plotted as points overlaying
    # aggregate data bars.
    grp_policies = calc_conditional_probs(trials,
                                          htrials,
                                          pred_col=pred_col,
                                          add_grps=grp,
                                          sortby='history',
                                          order=x_histories)
    fig, ax = plot_sequence_points(grp_policies, fig=fig, ax=ax, grp=grp,
                                   **kwargs)

    return fig, ax, grp_policies


def plot_block_seq_overview(trials, sortby='seq2', x='iInBlock',
                            block_length=None, multiple='stack', **kwargs):

    xlabel_lut = {'iInBlock': 'Block position',
                  'iBlock': 'Block in session'}

    # Use max block length found in data if no cutoff provided

    if block_length is None:
        # Min num trials per block pos, evaluated only to set upper limit.
        min_trials = kwargs.pop('min_trials', 1)
        block_length = int(trials.groupby(x, observed=True)
                                 .filter(lambda v: len(v) > min_trials)[x].max())
    trials_ = trials.query(f'{x}.between(0, @block_length)').sort_values(by=sortby)

    if kwargs.get('ax', None) is None:
        fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    else:
        ax = kwargs.pop('ax')
        fig = None
    plt.gcf().set_constrained_layout(False)

    sns.histplot(data=trials_, x=x, hue='seq2', ax=ax, stat='proportion',
                 bins=range(block_length + 2), multiple=multiple, linewidth=0,
                 **kwargs)
    ax.set(xticks=np.arange(block_length + 1, step=5),
           xlabel=xlabel_lut.get(x, x))
    leg = ax.get_legend()
    if leg:
        # Extract handles and labels
        handles = leg.legendHandles
        labels = [text.get_text() for text in leg.get_texts()]
        
        leg.remove()
        for handle in handles:
            handle.set_height(4)  # Smaller number = smaller height
            handle.set_width(6) 

        ax.legend(handles=handles, 
                labels=labels,
                bbox_to_anchor=(1.2, 0),
                title='trial type', 
                loc='center left',
                fontsize=9,
                labelspacing=0.2,  # Vertical spacing
                handletextpad=0.4)  # Horizontal spacing
    sns.despine()

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
    print(session_id)
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
    if bidx[-1] != tstop:
        bidx = np.insert(bidx, len(bidx), tstop)
    assert all(np.diff(bidx)) > 0, (
        'block transitions must monotonically increase')

    fig, ax = plt.subplots(figsize=[4.7, 1.2])
    first_state = session.State.iloc[0].item()
    # Iterate over blocks and shade odd blocks only (left blocks).
    for bstart, bstop in zip(bidx[:-1], bidx[1:]):
        # bstop is first trial of next block, gets offset below.
        # Skip blocks where State == 1 to "shade" them white.
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
    plt.yticks([0.05, 0.95], ['right', 'left  '], va='center')
    y = ax.get_yticklabels()
    y[1].set_backgroundcolor(shade_color)
    y[0].set_bbox(dict(facecolor='none', edgecolor='black'))
    ax.tick_params(axis='y', which='both', length=0)
    handles, labels = ax.get_legend_handles_labels()
    trial_type = {'0.8': 'unrewarded', '1.6': 'rewarded', 'timeout': 'timeout'}
    ax.legend(handles=handles, labels=[trial_type[label] for label in labels],
              bbox_to_anchor=(1, 1), loc='upper left', frameon=False,
              edgecolor=None, title='trial type')

    return fig, ax, session_id


def calc_bpos_probs(trials: pd.DataFrame,
                    add_agg_cols: list | str = None,
                    add_cond_cols: list | str = None) -> pd.DataFrame:

    '''
    Calculate probabilities of events conditioned on relative trial position
    in a block. Block transitions occur at `iInBlock` = 0.
    Args:
        trials:
            Trial-based data.
        add_agg_cols:
            Additional columns to calculate conditional probabilities for.
            Defaults to only calculating for Switch and High Port probability.
        add_cond_cols:
            Additional columns on which to condition probabilities. Defaults
            to only conditioning on block position.
    Returns:
        bpos:
            Dataframe containing P(agg_cols | cond_cols) for each trial in a
            block as `iInBlock` relative to block transitions at 0. Negative
            iInBlock corresponds to trial position relative to next transition.
    '''

    agg_cols = ['Switch', 'selHigh']
    if add_agg_cols is not None:
        if not isinstance(add_agg_cols, list):
            add_agg_cols = [add_agg_cols]
        agg_cols.extend(add_agg_cols)
    agg_funcs = {agg_col: 'mean' for agg_col in agg_cols}

    # Add column tracking how far trial is from end of block.
    trials_ = trials.copy()
    trials_['rev_iInBlock'] = trials_['iInBlock'] - trials_['blockLength']

    grp_forward = ['iInBlock']
    grp_rev = ['rev_iInBlock']
    if add_cond_cols is not None:
        if not isinstance(add_cond_cols, list):
            add_cond_cols = [add_cond_cols]
        grp_forward.extend(add_cond_cols)
        grp_rev.extend(add_cond_cols)

    bpos_probs = (trials_
                  .groupby(grp_forward, as_index=False, observed=True)
                  .agg(agg_funcs))
    bpos_probs_rev = (trials_
                      .groupby(grp_rev, as_index=False, observed=True)
                      .agg(agg_funcs))

    # Combine negative block positions with forward-counting positions.
    bpos_probs_rev = bpos_probs_rev.rename(columns={'rev_iInBlock': 'iInBlock'})
    bpos = (pd.concat((bpos_probs_rev, bpos_probs))
            .sort_values(by='iInBlock')
            .reset_index(drop=True))

    return bpos


def plot_bpos_behavior(bpos_probs: pd.DataFrame,
                       include_units: str = '',
                       plot_features: dict = None,
                       **kwargs):

    '''
    Plot mean probabilities relative to block transitions.
    Args:
        bpos_probs:
            Probabilities of events conditioned on relative trial position in
            block.
        include_units:
            Whether and which "units" (individual traces) to include alongside
            pooled trace.
        plot_features:
            Key, value pairs as (metric: (x-axis label, y-axis limits) for
            each subplot to include.
    Returns:
        figs, axs:
            Matplotlib objects containing plots.
    '''

    if plot_features is None:
        plot_features = {'selHigh': ('P(high port)', (0, 1)),
                         'Switch': ('P(switch)', (0, 0.4))}

    n_plots = len(plot_features.keys())

    if kwargs.get('ax', None) is None:
        fig, axs = plt.subplots(ncols=n_plots, figsize=(2.6 * n_plots, 1.8),
                                layout='constrained')
    else:
        axs = kwargs.pop('ax')
        fig = kwargs.pop('fig')
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for i, (metric, ax_vars) in enumerate(plot_features.items()):
        if include_units:
            if kwargs.get('cmap', False):
                plot_args = {'palette': kwargs.pop('cmap'),
                             'hue': include_units,
                             'legend': False
                             }
            else:
                plot_args = {'color': 'k',
                             'label': include_units}
            sns.lineplot(bpos_probs, x='iInBlock', y=metric, ax=axs[i],
                         units=include_units, estimator=None, lw=kwargs.get('lw', 0.6),
                         **plot_args)

        # When just plotting grand mean.
        if not kwargs.get('hue', False):
            sns.lineplot(bpos_probs, x='iInBlock', y=metric, ax=axs[i],
                         color='k', lw=2, label='pooled', **kwargs)
        else:
            sns.lineplot(bpos_probs, x='iInBlock', y=metric, ax=axs[i],
                         lw=2, **kwargs)

        label, ylim = ax_vars
        axs[i].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        axs[i].set(xlabel='block position', xlim=(-10, 10),
                   ylabel=label, ylim=ylim)

    sns.despine()
    axs[0].legend().remove()
    axs[-1].legend(bbox_to_anchor=(1.0, 1), edgecolor='white')
    check_leg_duplicates(axs[-1], coords=(1.0, 1))

    return fig, axs
