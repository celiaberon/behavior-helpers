import numpy as np
import pandas as pd


def map_licks(trials, licks):

    state_lut = {
        'cueP': [38, 39, 40],
        'enlP': [29, 30, 31],
        'select': [47, 48, 49],
        'select_end': [50, 51]
    }

    # Find cue penalties.
    CueP_data = trials.copy().query('n_Cue > 1')
    CueP_trials = CueP_data['nTrial'].values.astype('int')
    CueP_reps = CueP_data['n_Cue'].values.astype('int') - 1

    licks['selection_lick'] = 0

    if len(CueP_trials):
        # Repeat trial number by the number of CueP for that trial.
        all_CueP_trials = np.repeat(CueP_trials, CueP_reps)
        licks.loc[licks['sTrial_start'].isin(state_lut.get('cueP')), 'nTrial'] = all_CueP_trials

    # find ENLp (and account for +n_ENL when cue penalty but no ENL penalty)
    ENLP_data = trials.copy().query('n_ENL > 1 & n_ENL > n_Cue')
    ENLP_trials = ENLP_data['nTrial'].values.astype('int')
    ENLP_reps = (ENLP_data['n_ENL'] - ENLP_data['n_Cue']).values.astype('int')

    # Repeat trial number by the number of ENLp for that trial.
    all_ENLP_trials = np.repeat(ENLP_trials, ENLP_reps)
    licks.loc[licks['sTrial_start'].isin(state_lut.get('enlP')), 'nTrial'] = all_ENLP_trials

    # Selection state transition is usually 48-50, but can be [47, 48, 49] -> [50, 51].
    selection_data = trials.copy().query('timeout == False')
    all_selection_trials = selection_data['nTrial'].values
    selection_licks = licks.query(f'sTrial_start.isin({state_lut.get("select")})\
                                  & sTrial_end.isin({state_lut.get("select_end")})')
    print(len(selection_licks))
    print(len(all_selection_trials))

    if len(selection_licks) < len(all_selection_trials):
        missed_licks = licks.query('sTrial_start == 50 & sTrial_end == 51 & tState == 0')
        # Fully check for cases where the selection lick is the 50-51 transition first.
        for idx, row in missed_licks.iterrows():

            # If potential missed lick doesn't follow a selection state, this
            # confirms it is a missed lick.
            if idx == licks.iloc[0].name:
                print(idx)
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
                missed_licks = missed_licks.drop(idx)
                # continue
            elif (licks.loc[idx - 1, 'sTrial_end'] not in [47, 48, 49, 50]):
                print(idx)
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
                missed_licks = missed_licks.drop(idx)
                # continue

    if len(selection_licks) < len(all_selection_trials):
        # Then go back through for any unmapped 50-51 transitions and check again.
        for idx, row in missed_licks.iterrows():
            # Otherwise, it could still be a missed lick in rare cases where there was not a
            # consumption lick or ENLP following the previous selection lick.
            match_times = []
            for l_offset in np.arange(-4, 4):
                n_licks = len(selection_licks.loc[:idx]) + l_offset # licks up to possible missed lick
                try:
                    trial_id = all_selection_trials[n_licks]
                    selection_time = trials.query('nTrial == @trial_id')['tSelection'].squeeze()
                    target_time = selection_licks.iloc[n_licks].tState
                    match_times.append(selection_time == target_time)
                except IndexError:  # at session boundaries
                    # At this point, do we need another lick? Can't check alignment forward.
                    match_times.append(len(selection_licks) == len(all_selection_trials))

            # Times should match up until a missed lick, and then all will be
            # mismatched.
            print(match_times)
            match_times_int = [int(t) for t in match_times]
            if (np.sum(np.diff(match_times_int) == 1) == 0) and (not all(match_times)):
            # if (np.mean(match_times) < 0.3) and (match_times[5] == False):  # [True, False, False]):
                print('inserting', idx)
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
    print(len(selection_licks))
    licks.loc[selection_licks.index.values, 'nTrial'] = all_selection_trials
    licks.loc[selection_licks.index.values, 'selection_lick'] = 1

    # Fill forward with trial label.
    licks['nTrial'] = licks['nTrial'].ffill(axis=0)

    # Label blocks in lick_df using trials.
    trials = trials.set_index('nTrial')
    licks['iBlock'] = licks['nTrial'].map(trials['iBlock'])
    trials = trials.reset_index()

    return trials, licks


def get_selection_licks(trials, licks):

    state_lut = {
        'select': [47, 48, 49],
        'select_end': [50, 51]
    }

    licks['selection_lick'] = 0

    # Selection state transition is usually 48-50, but can be [47, 48, 49] -> [50, 51].
    selection_data = trials.copy().query('timeout == False')
    all_selection_trials = selection_data['nTrial'].values
    selection_licks = licks.query(f'sTrial_start.isin({state_lut.get("select")})\
                                  & sTrial_end.isin({state_lut.get("select_end")})')

    # Fully check for cases where the selection lick is the 50-51 transition first.
    if len(selection_licks) < len(all_selection_trials):
        missed_licks = licks.query('sTrial_start == 50 & sTrial_end == 51 & tState == 0')
        for idx, row in missed_licks.iterrows():
            # If potential missed lick doesn't follow a selection state, this
            # confirms it is a missed lick.
            if idx == licks.iloc[0].name:
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
                missed_licks = missed_licks.drop(idx)
            elif (licks.loc[idx - 1, 'sTrial_end'] not in [47, 48, 49, 50]):
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
                missed_licks = missed_licks.drop(idx)

    # Then go back through for any unmapped 50-51 transitions and check again.
    if len(selection_licks) < len(all_selection_trials):
        for idx, row in missed_licks.iterrows():
            # Otherwise, it could still be a missed lick in rare cases where there was not a
            # consumption lick or ENLP following the previous selection lick.
            match_times = []
            for l_offset in np.arange(-4, 4):
                n_licks = len(selection_licks.loc[:idx]) + l_offset # licks up to possible missed lick
                try:
                    trial_id = all_selection_trials[n_licks]
                    selection_time = trials.query('nTrial == @trial_id')['tSelection'].squeeze()
                    target_time = selection_licks.iloc[n_licks].tState
                    match_times.append(selection_time == target_time)
                except IndexError:  # at session boundaries
                    # At this point, do we need another lick? Can't check alignment forward.
                    match_times.append(len(selection_licks) == len(all_selection_trials))

            # Times should match up until a missed lick, and then all will be
            # mismatched.
            match_times_int = [int(t) for t in match_times]
            if (np.sum(np.diff(match_times_int) == 1) == 0) and (not all(match_times)):
                selection_licks = pd.concat((selection_licks, pd.DataFrame(row).T), axis=0).sort_index()
    licks.loc[selection_licks.index.values, 'selection_lick'] = 1

    return licks
