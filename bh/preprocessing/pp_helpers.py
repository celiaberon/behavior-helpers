import numpy as np
import pandas as pd


def single_session(func):
    def inner(*args, **kwargs):

        trials = args[0]
        assert trials.Session.dropna().nunique() == 1, (
            'Not written for multi-session processing')
        output = func(*args, **kwargs)
        return output
    return inner


@single_session
def get_block_pos(trials, dab=False):

    trials_ = trials.copy()
    if all(trials_['iBlock'].eq(0)):
        trials_['iBlock'] = trials_.DAB_I_flipLR_event.cumsum().astype('int') + 1
        trials_ = trials_.drop(columns=['DAB_I_flipLR_event'])

    trials_['iInBlock'] = trials_.groupby('iBlock').cumcount()

    return trials_


@single_session
def get_block_length(trials):

    trials_ = trials.copy()
    trials_['blockLength'] = trials_['iBlock'].map(trials_.value_counts('iBlock'))

    return trials_


@single_session
def define_switches(trials):

    '''
    Note on NaNs - NaNs, no selection, force a reset such that the next
    response trial cannot be defined as Switch or Stay (n+1 = NaN).
    '''
    trials_ = trials.copy()
    assert all(trials.loc[trials['timeout'] == True, 'direction'].isna())
    trials_['Switch'] = np.abs(trials['direction'].diff())

    return trials_


@single_session
def flag_blocks_for_timeouts(trials, threshold=0.25):

    trials_ = trials.copy()

    # Flag first and last block, and recursively flag n-1 continuous blocks
    # of timeouts > threshold.
    trials_['flag_block'] = False
    trials_.loc[trials_.iBlock == trials_.iBlock.min(), 'flag_block'] = True
    trials_.loc[trials_.iBlock == trials_.iBlock.max(), 'flag_block'] = True

    block_search = trials_.iBlock.max()-1
    continue_search = True
    while continue_search:
        curr_block = trials_.query('iBlock == @block_search')
        # If timeouts exceed threshold for timeouts
        if np.mean(curr_block.sSelection == 3) > threshold:
            trials_.loc[trials_.iBlock == block_search, 'flag_block'] = True
            block_search -= 1
        else:
            continue_search = False

    # also report on any block with timeouts above threshold
    # ([)code can likely be made more efficient by doing this first and then
    # finding from it flagged blocks instead of doing recursive search...)
    trials_['timeout_block'] = False
    for i, block in trials_.groupby('iBlock'):
        above_thresh = np.mean(block.sSelection == 3) > threshold
        trials_.loc[trials_.iBlock == i, 'timeout_block'] = above_thresh

    # record the threshold being used.
    trials_['timeout_thresh'] = threshold

    return trials_


def get_direction(df):

    """function that adds columns to report if the mouse made a choice to the right (1) or left (0) port"""

    target = df.State.values
    to_target = df.sSelection.values.astype(float)
    df['direction'] = 1 - ((to_target - 1) == target).astype('float')
    df.loc[df.sSelection == 3, 'direction'] = np.nan

    return df