import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(f'{os.path.expanduser("~")}/GitHub/neural-timeseries-analysis/')

import nta.preprocessing.quality_control as qc
from nta.data.datasets import DataSet
from nta.features import behavior_features as bf
from nta.utils import load_config_variables


class HFDataSet(DataSet):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):

        super().__init__(mice, **kwargs)
        self.qc_photo = None  # behavior only
        self.channels = None  # behavior only
        self.sig_channels = None  # behavior only
        self.cohort = None
        self.palettes = self.load_color_palettes()

    def set_root(self):
        '''Sets the root path for the dataset'''
        return Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/')

    def set_data_path(self):
        '''Sets the path to the session data'''
        return self.root / 'headfixed_DAB_data/preprocessed_data'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        return self.root / 'data_overviews' / 'session_log_all_cohorts.csv'

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_

    def set_save_path(self):
        '''Set save path and create the directory.'''
        save_path = self.root / 'headfixed_DAB_data/behav_figures' / self.label
        if not os.path.exists(os.path.join(save_path, 'metadata')):
            os.makedirs(os.path.join(save_path, 'metadata'))
        return save_path

    def load_cohort_dict(self):
        pass

    def load_color_palettes(self):

        '''Load standard color palettes for plotting'''
        palettes = load_config_variables(self.root)
        return palettes

    def update_columns(self, trials, ts):

        '''
        Column updates (feature definitions, etc.) that should apply to all
        datasets.
        '''
        trials, ts = bf.add_behavior_cols(trials, ts)
        trials = trials.rename(columns={'-1reward': 'prev_rew'})

        # Rectify error in penalty state allocation.
        ts['ENL'] = ts['ENL'] + ts['state_ENLP']  # recover original state
        ts['Cue'] = ts['Cue'] + ts['CueP']  # recover original state
        ts = bf.split_penalty_states(ts, penalty='ENLP')
        ts = bf.split_penalty_states(ts, penalty='CueP')

        return trials, ts

    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        file_path = self.set_session_path()
        ts_path = file_path / f'{self.mouse_}_analog_filled.csv'
        return ts_path

    def load_session_data(self):
        '''Loads data from single session'''
        trials_path = self.set_trials_path()
        ts_path = self.set_timeseries_path()

        if not (ts_path.exists() & trials_path.exists()):
            if self.verbose:
                print(f'skipped {self.mouse_} {self.session_}')
            return None, None

        trial_dtypes = {
            'nTrial': np.int32,
            'Mouse': 'object',
            'Date': 'object',
            'Session': 'object',
            'Condition': np.int16,
            'tSelection': np.int16,
            'direction': np.float32,
            'Reward': np.float32,
            'T_ENL': np.int16,
            'n_ENL': np.int8,
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

        usecols = list(trial_dtypes.keys())
        trials = pd.read_csv(trials_path, index_col=None, dtype=trial_dtypes,
                             usecols=usecols)

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
            'trial_clock': 'float'
        }

        usecols = list(ts_dtypes.keys())
        ts = pd.read_csv(ts_path, index_col=None, usecols=usecols, dtype=ts_dtypes)

        return ts, trials

    def read_multi_sessions(self,
                            qc_params,
                            **kwargs) -> dict:

        sessions = self.sessions_to_load(**qc_params)
        multi_sessions = {key: pd.DataFrame() for key in ['trials', 'ts']}

        # Loop through files to be processed
        for session_date in tqdm(sessions, self.mouse_, disable=False):

            if self.verbose: print(session_date)
            self.session_ = session_date

            ts, trials = self.load_session_data()
            if ts is None: continue
            trials, ts = self.update_columns(trials, ts)

            # Trial level quality control needs to come at the end.
            trials_matched = qc.QC_included_trials(ts,
                                                   trials,
                                                   allow_discontinuity=False,
                                                   drop_enlP=False)

            multi_sessions = self.concat_sessions(sub_sessions=trials_matched,
                                                  full_sessions=multi_sessions)

            n_sessions_mouse = (multi_sessions['trials']
                                .query(f'Mouse == {self.mouse_} & nTrial > 1')['Session']
                                .nunique())
            if (self.session_cap is not None) and (n_sessions_mouse >= self.session_cap):
                break
        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        gc.collect()

        return multi_sessions


class HFTrials(HFDataSet):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):

        super().__init__(mice, **kwargs)

    def update_columns(self, trials):

        '''
        Column updates (feature definitions, etc.) that should apply to all
        datasets.
        '''
        trials = bf.add_behavior_cols(trials)
        # trials = trials.rename(columns={'-1reward': 'prev_rew'})

        return trials

    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        pass

    def load_session_data(self):
        '''Loads data from single session'''
        trials_path = self.set_trials_path()

        if not trials_path.exists():
            if self.verbose: print(f'skipped {self.mouse_} {self.session_}')
            return None, None

        trials = pd.read_csv(trials_path, index_col=0)

        return trials

    def get_max_trial(self, full_sessions: dict) -> int:

        '''
        Get maximum trial ID to use for unique trial ID assignment.
        Importantly, also confirm that max trial matches between dataframes.

        Args:
            full_sessions:
                Dictionary containing and trial- and timeseries-based data.

        Returns:
            max_trial:
                Number corresponding to maximum trial value.
        '''

        try:
            max_trial_trials = full_sessions['trials'].nTrial.max()
            max_trial = max_trial_trials
        except AttributeError:
            max_trial = 0

        return max_trial

    def read_multi_mice(self,
                        qc_params,
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
                {'trials': trials data, 'timeseries': timeseries data}
        '''

        multi_mice = {key: pd.DataFrame() for key in ['trials']}

        for mouse in self.mice:

            self.mouse_ = mouse

            multi_sessions = self.read_multi_sessions(qc_params, **kwargs)

            if len(multi_sessions.get('trials')) < 1:
                continue  # skip mouse if no sessions returned

            multi_mice = self.concat_sessions(sub_sessions=multi_sessions,
                                              full_sessions=multi_mice)

        self.trials = multi_mice.get('trials')
        print(f'{self.trials.Session.nunique()} total sessions loaded in')

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

            trials = self.update_columns(trials)
            multi_sessions = self.concat_sessions(sub_sessions={'trials': trials},
                                                  full_sessions=multi_sessions)

        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        return multi_sessions

    def get_sampling_freq(self):

        '''Lazy workaround for inheriting from timeseries dataset'''
        pass
