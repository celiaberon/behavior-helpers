import gc
# import getpass
import os
import re
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(f'{os.path.expanduser("~")}/GitHub/neural-timeseries-analysis/')

import nta.preprocessing.quality_control as qc
from nta.features import behavior_features as bf

from ..utils import (cast_object_to_category, convert_path_by_os,
                     downcast_all_numeric, load_config_variables)


class HFTrials(ABC):

    def __init__(self,
                 mice: str | list[str],
                 user: str = 'celia',
                 verbose: bool = False,
                 label: str = '',
                 save: bool = True,
                 qc_params: dict = {},
                 add_cols: dict[set] = {},
                 session_cap: int = None,
                 ):

        self.mice = mice
        self.user = user
        self.verbose = verbose
        self.label = label if label else self.mice
        self.session_cap = session_cap  # max number of sessions per mouse
        self.qc_params = qc_params
        self.trls_add_cols = add_cols.get('trials', set())

        # Set up paths and standard attributes.
        self.root = self.set_root()
        self.config_path = self.set_config_path()
        self.data_path = self.set_data_path()
        self.summary_path = self.set_data_overview_path()
        self.save = save
        if self.save:
            self.save_path = self.set_save_path()
        self.cohort = self.load_cohort_dict()
        self.palettes = load_config_variables(self.config_path)
        self.add_mouse_palette()

        self.trials = pd.DataFrame()
        self.max_trial = 0

    def load_data(self):

        # Load all data.
        if not isinstance(self.mice, list):
            self.mouse_ = self.mice
            multi_sessions = self.read_multi_sessions(self.qc_params)
            self.trials = multi_sessions.get('trials')
        else:
            multi_mice = self.read_multi_mice(self.qc_params, df_keys=['trials'])
            # Store data from multi_mice as attributes of dataset.
            self.trials = multi_mice.get('trials')

        print(f'{self.trials.Session.nunique()} total sessions loaded in')

        if 'session_order' in self.session_log:
            self.trials = bf.order_sessions(self.trials, self.session_log)
        else:
            self.trials = bf.order_sessions(self.trials)

        # Downcast datatypes to make more memory efficient.
        self.downcast_dtypes()

        gc.collect()

    @abstractmethod
    @convert_path_by_os
    def set_root(self):
        '''Sets the root path for the dataset'''
        pass

    @abstractmethod
    @convert_path_by_os
    def set_config_path(self):
        '''Sets the path to config file'''
        pass

    @abstractmethod
    @convert_path_by_os
    def set_data_path(self):
        '''Sets the path to the session data'''
        pass

    @abstractmethod
    @convert_path_by_os
    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        pass

    @convert_path_by_os
    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_

    @convert_path_by_os
    def set_trials_path(self):
        '''Set path to trial-level data file.'''
        file_path = self.set_session_path()
        trials_path = file_path / f'{self.mouse_}_trials.csv'
        return trials_path

    @convert_path_by_os
    def set_save_path(self):
        '''Set save path and create the directory.'''
        save_path = self.root / 'headfixed_DAB_data/behav_figures' / self.label
        if not os.path.exists(self.set_metadata_path()):
            os.makedirs(self.set_metadata_path())
        return save_path
    
    @convert_path_by_os
    def set_metadata_path(self):

        return self.root / 'headfixed_DAB_data/behav_figures' / self.label / 'metadata'

    def load_cohort_dict(self):
        '''Load lookup table for sensor expressed in each mouse of cohort.'''
        cohort = load_config_variables(self.config_path, 'cohort')['cohort']
        cohort = ({k: cohort.get(k) for k in self.mice} 
                  if isinstance(self.mice, list)
                  else {self.mice: cohort.get(self.mice)})
        return cohort

    def add_mouse_palette(self):
        '''Set up some consistent mapping to distinguish mice in plots.'''
        pal = sns.color_palette('deep', n_colors=len(self.mice))
        if isinstance(self.mice, list):
            self.palettes['mouse_pal'] = {mouse: color for mouse, color
                                          in zip(self.mice, pal)}
        else:
            self.palettes['mouse_pal'] = {self.mice: pal[0]}

    def sessions_to_load(self,
                         probs: int | str = 9010,
                         QC_pass: bool = True,
                         **kwargs) -> list:

        '''
        Make list of sessions to include for designated mouse

        Args:
            probs:
                Filter bandit data by probability conditions.
            QC_pass:
                Whether to take sessions passing quality control (True) or
                failing (False).

        Returns:
            dates_list:
                List of dates to load in for mouse, sorted from earliest to
                latest.
        '''

        # Read in session log.
        session_log = pd.read_csv(self.summary_path)
        self.session_log = session_log

        # Format inputs/args correctly.
        if not isinstance(QC_pass, list):
            QC_pass = [QC_pass]
        if isinstance(probs, list):
            probs = [str(p) for p in probs]
        else:
            probs = [str(probs)]

        # Compose query.
        session_log_mouse = session_log.query(f'Mouse == "{self.mouse_}" \
                                              & Condition.isin({probs})')
        q = f'Mouse == "{self.mouse_}" & Condition.isin({probs}) \
            & N_valid_trials > {kwargs.get("min_num_trials", 100)} \
            & Pass.isin({QC_pass})' + kwargs.get("query", '')
        session_log = session_log.query(q)
        if self.verbose:
            print(f'{self.mouse_}: {len(session_log)} of',
                  f' {len(session_log_mouse)} sessions meet criteria')
        return sorted(list(set(session_log.Date.values)))

    def define_trial_dtypes(self):

        trial_dtypes = {
            'nTrial': np.int32,
            'Mouse': 'object',
            'Date': 'object',
            'Session': 'object',
            'Condition': 'object',
            'tSelection': np.int16,
            'direction': np.float32,
            'Reward': np.float32,
            'T_ENL': np.int16,
            'n_ENL': np.int16,
            'n_Cue': np.int8,
            'State': np.float32,
            'selHigh': np.float32,
            'iBlock': np.int8,
            'blockLength': np.int8,
            'iInBlock': np.int8,
            'flag_block': 'bool',
            'timeout': 'bool',
            'Switch': np.float32
        }

        return trial_dtypes

    def load_data_flex_cols(self, data_path, data_types, usecols):
        '''Load timeseries data but be forgiving about missing columns.'''
        while usecols:
            try:
                if data_path.suffix == '.csv':
                    data = pd.read_csv(data_path, index_col=None,
                                       usecols=usecols, dtype=data_types)
                elif data_path.suffix == '.gzip':
                    data = (pd.read_parquet(data_path, columns=usecols)
                            .astype(data_types))
                # Create session column to match across dataframes.
                if 'session' in data.columns:
                    data = data.rename(columns={'session': 'Session'})
                return data

            except ValueError as e:
                # Extract the missing column name from the error message.
                re_match = [
                    re.search(r'\((.*?)\)', str(e)),
                    re.search(r'"(.*?)"', str(e)),
                    re.search(r"'(.*?)'", str(e))
                ]

                re_match = [s for s in re_match if s is not None]
                for s in re_match:
                    if isinstance(s, re.Match) and (s.group(1) in usecols):
                        missing_col = s.group(1)
                        if missing_col == 'session':
                            data_types['Session'] = data_types.pop('session')
                            usecols.append('Session')
                        usecols.remove(missing_col)
                        data_types.pop(missing_col, None)
                        break
                else:
                    # In the case we can't find missing column.
                    raise e
        raise ValueError('All specified columns missing from data file.')

    def load_trial_data(self):
        '''Loads trial data from single session'''
        trials_path = self.set_trials_path()
        if not trials_path.exists():
            if self.verbose:
                print(f'skipped {self.mouse_} {self.session_}')
            return None

        trial_dtypes = self.define_trial_dtypes()

        usecols = list(trial_dtypes.keys()) + list(self.trls_add_cols)
        trials = self.load_data_flex_cols(trials_path, trial_dtypes, usecols)
        return trials

    def load_session_data(self):
        return self.load_trial_data()

    def get_max_trial(self,
                      full_sessions: dict,
                      keys=['trials', 'ts']) -> int:

        '''
        Get maximum trial ID to use for unique trial ID assignment.
        Importantly, also confirm that max trial matches between dataframes.

        Args:
            full_sessions:
                Dictionary containing and trial- and possibly timeseries-based data.

        Returns:
            max_trial:
                Number corresponding to maximum trial value.
        '''

        try:
            max_df_trial = [full_sessions[key].nTrial.max() for key in keys]
            if len(max_df_trial) > 1:
                assert np.allclose(*max_df_trial, rtol=0)
            max_trial = max_df_trial[0]
        except AttributeError:
            max_trial = 0
        return max_trial

    def at_session_cap(self, multi_sessions):

        ''''
        Check whether number of sessions for a given mouse has reached a given
        session cap (max number of sessions per mouse to load). If no session
        cap provided, return false and load all data.
        '''

        if self.session_cap is None:
            return False

        if multi_sessions['trials'].Session.nunique() >= self.session_cap:
            return True

        return False

    def concat_sessions(self,
                        *,
                        sub_sessions: dict = None,
                        full_sessions: dict = None):

        '''
        Aggregate multiple sessions by renumbering trials to provide unique
        trial ID for each trial. Store original id in separate column.

        Args:
            sub_sessions:
                Smaller unit to be concatenated onto growing aggregate df.
            full_sessions:
                Core larger unit updated with aggregating data.

        Returns:
            full_sessions:
                Original full_sessions data now containing sub_sessions data.
        '''
        self.max_trial = self.get_max_trial(full_sessions,
                                            keys=list(full_sessions.keys()))
        # Iterate over both trial and timeseries data.
        for key, ss_vals in sub_sessions.items():
            # Store original trial ID before updating with unique value.
            if 'nTrial_orig' not in ss_vals.columns:
                ss_vals['nTrial_orig'] = ss_vals['nTrial'].copy()

            # Create session column to match across dataframes.
            if 'Session' not in ss_vals.columns:
                ss_vals['Session'] = '_'.join([self.mouse_, self.session_])

            # Add max current trial value to all new trials before concat.
            tmp_copy = ss_vals.copy()
            tmp_copy['nTrial'] += self.max_trial
            full_sessions[key] = pd.concat((full_sessions[key], tmp_copy))
            full_sessions[key] = full_sessions[key].reset_index(drop=True)

        assert len(full_sessions['trials']) == full_sessions['trials'].nTrial.nunique(), (
                    'each trial does not have a unique ID'
                )

        # # Assert that new dataframes have matching max trial ID.
        # _ = self.get_max_trial(full_sessions, keys=list(full_sessions.keys()))

        return full_sessions

    def read_multi_mice(self,
                        qc_params,
                        df_keys: list[str] = ['trials', 'ts'],
                        **kwargs):

        '''
        Load in sessions by mouse and concatenate into one large dataframe
        keeping every trial id unique.

        Args:
            mice:
                List of mice from which to load data.
            root:
                Path to root directory containing Mouse data folder.

        Returns:
            multi_mice (dict):
                {'trials': trials data, 'ts': timeseries data}
        '''

        multi_mice = {key: pd.DataFrame() for key in df_keys}

        for mouse in self.mice:
            self.mouse_ = mouse
            multi_sessions = self.read_multi_sessions(qc_params, **kwargs)

            if len(multi_sessions.get('trials')) < 1:
                continue  # skip mouse if no sessions returned
            multi_mice = self.concat_sessions(sub_sessions=multi_sessions,
                                              full_sessions=multi_mice)

        sig_mice = multi_mice['ts'].Mouse.unique().tolist()
        sig_mice = set(sig_mice) if isinstance(self.mice, list) else set([sig_mice])
        if set(self.mice) != sig_mice:
            print(f'mice {set(self.mice) - sig_mice} have no sessions')
        self.mice = sig_mice
        # self.trials = multi_mice.get('trials')
        return multi_mice

    def read_multi_sessions(self,
                            qc_params,
                            **kwargs) -> dict:

        sessions = self.sessions_to_load(**qc_params)
        multi_sessions = {key: pd.DataFrame() for key in ['trials']}

        # Loop through files to be processed
        for session_date in tqdm(sessions, self.mouse_, disable=False):

            if self.verbose: print(session_date)
            self.session_ = session_date

            trials = self.load_session_data()
            if trials is None: continue
            trials = self.custom_update_columns(trials)
            trials = self.update_columns(trials)
            trials = self.cleanup_cols(trials)
            multi_sessions = self.concat_sessions(sub_sessions={'trials': trials},
                                                  full_sessions=multi_sessions)


        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        return multi_sessions

    def update_columns(self, trials):

        '''
        Column updates (feature definitions, etc.) that should apply to all
        datasets.
        '''
        # Check for state labeling consistency.
        trials = bf.match_state_left_right(trials)
        trials = bf.add_behavior_cols(trials)
        # trials = trials.rename(columns={'-1reward': 'prev_rew'})

        return trials

    def custom_update_columns(self, trials):
        '''Column updates that are dataset-specific.'''
        return trials

    def cleanup_cols(self, dfs: dict | pd.DataFrame):
        '''Remove unnecessary columns to minimize memory usage.'''
        return dfs

    def downcast_dtypes(self):

        self.trials = downcast_all_numeric(self.trials)
        self.trials = cast_object_to_category(self.trials)


class HFDataset(HFTrials):

    '''
    Full headfixed dataset class (containing trials and timeseries data).
    '''

    def __init__(self,
                 mice: str | list[str],
                 add_cols: dict[set] = {},
                 **kwargs):

        super().__init__(mice, add_cols=add_cols, **kwargs)

        self.ts_add_cols = add_cols.get('ts', set())
        self.ts = pd.DataFrame()

    def load_data(self):

        # Load all data.
        if not isinstance(self.mice, list):
            self.mouse_ = self.mice
            multi_sessions = self.read_multi_sessions(self.qc_params)
            self.ts = multi_sessions.get('ts')
            self.trials = multi_sessions.get('trials')
        else:
            multi_mice = self.read_multi_mice(self.qc_params, df_keys=['trials', 'ts'])
            # Store data from multi_mice as attributes of dataset.
            self.ts = multi_mice.get('ts')
            self.trials = multi_mice.get('trials')
        print(f'{self.trials.Session.nunique()} total sessions loaded in')

        if 'session_order' in self.session_log:
            self.trials = bf.order_sessions(self.trials, self.session_log)
        else:
            self.trials = bf.order_sessions(self.trials)

        # Some validation steps on loaded data.
        self.check_event_order()

        # Downcast datatypes to make more memory efficient.
        self.downcast_dtypes()
        gc.collect()

    @convert_path_by_os
    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        file_path = self.set_session_path()
        ts_path = file_path / f'{self.mouse_}_analog_filled.csv'
        return ts_path

    def define_ts_dtypes(self):

        ts_dtypes = {
            'nTrial': np.float32,
            'iBlock': np.float32,
            'session': 'object',
            'session_clock': 'float',
            'iSpout': np.int8,
            'ENLP': np.int8,
            'CueP': np.int8,
            'ENL': np.int8,
            'Cue': np.int8,
            'Select': np.int8,
            'stateConsumption': np.int8,
            'TO': np.int8,
            'system_nTrial': np.float32,
            'outcome_licks': np.int8,
            'Consumption': np.int8,
            'state_ENLP': np.int8,
            'state_CueP': np.int8,
            'responseTime': np.int8,
            'state_ENL_preCueP': np.int8
            # 'trial_clock': 'float'
        }

        if self.user != 'celia':  # updated naming of session column
            ts_dtypes['Session'] = ts_dtypes.pop('session')

        return ts_dtypes

    def define_ts_cols(self):

        ts_dtypes = self.define_ts_dtypes()
        usecols = list(ts_dtypes.keys())
        usecols.extend(list(self.ts_add_cols))
        usecols = list(set(usecols))
        return ts_dtypes, usecols

    def load_ts_data(self):
        '''Loads data from single session'''
        ts_path = self.set_timeseries_path()

        if not ts_path.exists():
            if self.verbose:
                print(f'skipped {self.mouse_} {self.session_}')
            return None

        ts_dtypes, usecols = self.define_ts_cols()
        # Load timeseries data but be forgiving about missing columns.
        ts = self.load_data_flex_cols(ts_path, ts_dtypes, usecols)
        return ts

    def load_session_data(self):
        trials = self.load_trial_data()
        ts = self.load_ts_data()
        return trials, ts

    def read_multi_sessions(self,
                            qc_params,
                            **kwargs) -> dict:

        sessions = self.sessions_to_load(**qc_params)
        multi_sessions = {key: pd.DataFrame() for key in ['trials', 'ts']}

        # Loop through files to be processed
        for session_date in tqdm(sessions, self.mouse_, disable=False):

            if self.verbose: print(session_date)
            self.session_ = session_date

            trials, ts = self.load_session_data()
            if ts is None: continue
            trials, ts = self.custom_update_columns(trials, ts)
            trials, ts = self.update_columns(trials, ts)
            trials, ts = self.custom_dataset_pp(trials, ts, **kwargs)
            if ts is None: continue
            # Trial level quality control needs to come at the end.
            trials_matched = qc.QC_included_trials(ts,
                                                   trials,
                                                   allow_discontinuity=False)

            trials_matched = self.cleanup_cols(trials_matched)
            multi_sessions = self.concat_sessions(sub_sessions=trials_matched,
                                                  full_sessions=multi_sessions)

            if self.at_session_cap(multi_sessions): break

        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        gc.collect()

        return multi_sessions

    def check_event_order(self):

        '''
        Test to ensure no events appear to occur out of task-defined trial
        order. Note, it's expected that a few edge cases may be given the wrong
        trial ID. This seems to only happen for ENLPs that happen at trial time
        of 0 (assigned to preceding trial. They can be  dealt with, but any
        other cases should raise an alarm.
        '''

        trial_event_order = {
            'ENLP': 1,
            'state_ENLP': 1,
            'state_ENL_preCueP': 1,
            'CueP': 1,
            'state_CueP': 1,
            'ENL': 2,
            'Cue': 3,
            'responseTime': 4,
            'Select': 4,
            'stateConsumption': 5,
            'Consumption': 5,
            'TO': 4,
        }

        ts = self.ts.copy()
        ts['event_order'] = np.nan

        for event, val in trial_event_order.items():
            if event not in ts.columns: continue
            ts.loc[ts[event] == 1, 'event_order'] = val

        # Should be monotonic increase. Any events out of order get flagged.
        out_of_order = (ts.dropna(subset='event_order')
                        .groupby('nTrial', observed=True)['event_order']
                        .diff() < 0)

        ooo = ts.loc[out_of_order[out_of_order].index]
        ooo_trials = ooo.nTrial.unique()

        ts['flag_ooo'] = np.nan
        ts.loc[ooo.index, 'flag_ooo'] = 1

        # Forward fill flag_ooo column in ts with 1 for each nTrial that has an ooo event.
        ts['flag_ooo'] = ts.groupby('nTrial')['flag_ooo'].ffill(1)

        ooo_and_post = ts.query('flag_ooo==1')
        rows_ooo = len(ooo_and_post)

        assert (ooo_and_post['ENLP'].all()) & (not any(ooo_and_post[['Cue', 'Select', 'Consumption', 'ENL']].any())), (
                'events out of order beyond ENLP edge cases')

        # assert rows_ooo == len(ooo), (
        #     'rows out of order following ENLP edge cases')

        ts['nan_next_event'] = ts['event_order'].copy()
        ts['nan_next_event'] = ts['nan_next_event'].bfill()

        print(ts.query('event_order.isna()')['nan_next_event'].unique(), ts.query('event_order.isna()')['nan_next_event'].value_counts())

        # assert all(ts.query('event_order.isna()')['nan_next_event'].dropna().unique() == trial_event_order.get('ENL')), (
        #     'only unlabeled event is NOT transitioning into ENL')
        assert ((len(ts.query('event_order.isna()'))) < (100 * ts.Session.nunique())), (
            f'rate of unlabeled events is too high at {len(ts.query("event_order.isna()"))}')

        self.ts.loc[ooo_and_post.index, 'ENLP'] = np.nan

        # Cleanup columns a bit.
        cols_to_drop = {'state_ENL_preCueP', 'state_CueP', 'state_ENLP',
                        'stateConsumption', 'CueP', 'responseTime'}
        cols_to_drop = list(cols_to_drop - self.ts_add_cols)
        self.ts = self.ts.drop(columns=cols_to_drop)

    def custom_update_columns(self, trials, ts):
        '''Column updates that are dataset-specific.'''
        return trials, ts

    def update_columns(self, trials, ts):

        # Check for state labeling consistency.
        trials = bf.match_state_left_right(trials)

        # Add standard set of analysis columns.
        trials, ts = bf.add_behavior_cols(trials, ts)
        trials = trials.rename(columns={'-1reward': 'prev_rew'})

        # Rectify error in penalty state allocation.
        # ts['ENL'] = ts['ENL'] + ts['state_ENLP'] + ts.get('state_ENL_preCueP', 0)  # recover original state
        # ts['Cue'] = ts['Cue'] + ts['CueP']  # recover original state
        # ts = bf.split_penalty_states(ts, penalty='ENLP')
        # ts = bf.split_penalty_states(ts, penalty='CueP')
        # ts = bf.split_penalty_states(ts, penalty='CueP', cuep_ref_enl=True)

        if 'trial_clock' not in ts.columns:
            if 'PhotometryDataset' not in [b.__name__ for b in self.__class__.__bases__]:
                fs = 200
                assert np.allclose(fs, 1 / ts['session_clock'].diff()[1], atol=0.01)
                ts['fs'] = ts.get('fs', fs)
            ts['trial_clock'] = 1 / ts['fs']
            ts['trial_clock'] = ts.groupby('nTrial', observed=True)['trial_clock'].cumsum()
        else:
            if 'HFDataset' in [b.__name__ for b in self.__class__.__bases__]:
                # Behavior only timeseries ok if trial_clock already exists
                return trials, ts
            # print(ts.columns)
            assert ts['fs'].iloc[0] is not None, 'need sampling freq to add trial clock'
            ts['trial_clock'] = ts.groupby('nTrial').cumcount() * 1/ts['fs'].iloc[0]

        assert sum(ts['trial_clock'].diff() < 0) == (ts.nTrial.nunique() - 1), (
            'trial_clock is incorrect')

        return trials, ts

    def custom_dataset_pp(self, trials, ts, **kwargs):

        return trials, ts

    def cleanup_cols(self, df_dict):

        '''Remove unnecessary columns to minimize memory usage.'''

        # Drop columns that aren't typically accessed for analysis but were
        # necessary for preprocessing.
        cols_to_drop = {
            # 'state_ENL_preCueP', #'state_CueP', 'state_ENLP',
                        # 'stateConsumption', 'CueP',
                        'iLick', 'ILI',
                        'bout_group', 'cons_bout'
                        } & set(df_dict['ts'].columns)
        cols_to_drop = list(cols_to_drop - self.ts_add_cols)
        df_dict['ts'] = df_dict['ts'].drop(columns=cols_to_drop)

        cols_to_drop = {'k1', 'k2', 'k3', '+1seq2', 'RL_seq2', 'RL_seq3',
                        '-1seq3', '+1seq3',
                        } & set(df_dict['trials'].columns)
        col_to_drop = list(cols_to_drop - self.trls_add_cols)
        df_dict['trials'] = df_dict['trials'].drop(columns=col_to_drop)

        gc.collect()

        return df_dict

    def downcast_dtypes(self):

        '''
        Downcast columns in trial and timeseries dataframes by datatype if
        possible.
        '''
        self.trials = downcast_all_numeric(self.trials)
        self.ts = downcast_all_numeric(self.ts)
        self.ts = cast_object_to_category(self.ts)
        self.trials = cast_object_to_category(self.trials)
