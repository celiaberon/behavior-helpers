import os
import sys
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ssm
from sklearn.model_selection import train_test_split
from ssm.model_selection import cross_val_scores

sys.path.append(f'{os.path.expanduser("~")}/GitHub/neural-timeseries-analysis/')
from nta.features import behavior_features as bf

from bh.utils import calc_ci, make_onehot_array


class ModelData(ABC):

    def __init__(self):
        pass

    def drop_nans(self):
        pass

    def prepare_features(self, trials, yvar, feat_funcs=None, nlags=3):

        '''
        Create design matrix containing features (transformed) and history
        features up to nlags.
        '''
        self.nlags = nlags  # history length per feature

        def pm1(x):
            return 2 * x - 1

        if feat_funcs is None:
            feat_funcs = {'Reward': lambda r, c: r,
                          'direction': lambda r, c: pm1(c),
                          'dir-rew': lambda r, c: r * pm1(c)
                          }
        else:
            assert all([col in ['Reward', 'direction', 'dir-rew']
                        for col in feat_funcs]), (
                'Transformations only work for Reward- and Direction-based inputs.'
            )

        self.data_raw = trials.copy()
        initial_cols = [col for col in feat_funcs if col in trials.columns]

        # Just drop NaNs (timeouts) for now.
        trials_clean = trials.dropna(subset=['Reward', 'direction'])

        # Keep session and trial ID for now to register train/test splits.
        self.y = trials_clean[[yvar, 'Session', 'nTrial']].copy()
        self.X = trials_clean[initial_cols + ['Session', 'nTrial']].copy()

        # Forward and backward shifts (need to shift before any splitting).
        for feature, func in feat_funcs.items():

            # Apply transformation to the feature.
            self.X[feature] = func(trials_clean['Reward'].values,
                                   trials_clean['direction'].values)
            print(feature, self.X[feature].unique())
            for lag in range(1, nlags + 1):
                self.X = bf.shift_trial_feature(self.X,
                                                col=feature,
                                                n_shift=lag,
                                                shift_forward=True)

        assert all([col in self.X.columns for col in list(feat_funcs.keys())])

        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        # Get indices for rows containing NaNs in any features (including
        # lagged histories) and drop from feature matrix.
        lag_nulls = np.unique(np.where(np.isnan(self.X.drop(columns=['Session'])))[0])
        self.X = self.X.drop(index=lag_nulls)
        self.y = self.y.drop(index=lag_nulls)

        # Drop original feature columns and keep only transformed ones.
        self.X = self.X.drop(columns=list(feat_funcs.keys()))

        # Store list of feature columns names.
        self.features = self.X.columns.drop(['Session', 'nTrial']).values

    def get_data_subset(self, dataset='X', col='Session', vals=None):

        '''Query dataset for matching values in a particular column.'''
        if dataset == 'X':
            return (self.X.copy()
                    .query(f'{col}.isin(@vals)')
                    .reset_index(drop=True))

        else:
            return (self.y.copy()
                    .query(f'{col}.isin(@vals)')
                    .reset_index(drop=True))

    def iter_by_session(self, X, y):

        '''
        Create nested lists of len()=number of sessions, so that session
        becomes and iterable.
        '''
        # Index of first trial (of dataset) in each session.
        session_starts = (X
                          .groupby('Session', observed=True)
                          .nth(0)
                          .index.values)
        session_starts = np.concatenate((session_starts, [len(X)]))

        # All trial ids, by session in a nested list.
        trial_ids = [X.loc[start:stop - 1].nTrial.values
                     for start, stop in zip(session_starts[:-1], session_starts[1:])]
        X = X.drop(columns=['Session', 'nTrial'])
        y = y.drop(columns=['Session', 'nTrial'])

        # Convert datasets to nested lists over sessions.
        X_by_sess = [X.loc[start:stop - 1].to_numpy()
                     for start, stop in zip(session_starts[:-1], session_starts[1:])]
        y_by_sess = [y.loc[start:stop - 1].to_numpy().reshape(-1, 1).astype('int')
                     for start, stop in zip(session_starts[:-1], session_starts[1:])]

        return X_by_sess, y_by_sess, trial_ids

    def split_data(self, ptrain=0.8, seed=0, verbose=False):

        '''
        Split data into train and tests sets by session ID. Note, this severely
        limits the possible combinations of train/val/test datasets.
        '''
        # Assign session ids to train and test splits.
        train_ids, test_ids = train_test_split(self.X['Session'].unique(),
                                               train_size=ptrain,
                                               random_state=seed)

        if verbose:
            print(f'{len(train_ids)} training sessions and',
                  f'{len(test_ids)} test sessions')

        # Make sure id lists are complete and contain unique values only.
        assert len(train_ids) == len(set(train_ids))
        assert len(test_ids) == len(set(test_ids))

        self.train_num_sess = len(train_ids)
        self.test_num_sess = len(test_ids)

        # Split data into train and test sets.
        train_X = self.get_data_subset(dataset='X', col='Session',
                                       vals=train_ids)
        train_y = self.get_data_subset(dataset='y', col='Session',
                                       vals=train_ids)

        self.train_num_trials = len(train_X)
        assert len(train_y) == self.train_num_trials, (
            'Different number of trials in train X and y')

        # Store datasets with session as iterable first level.
        self.train_X, self.train_y, self.train_trials = self.iter_by_session(train_X, train_y)
        assert len(self.train_y) == len(self.train_X), (
            'Different number of sessions in train X and y')

        # Test set.
        test_X = self.get_data_subset(dataset='X', col='Session',
                                      vals=test_ids)
        test_y = self.get_data_subset(dataset='y', col='Session',
                                      vals=test_ids)

        self.test_num_trials = len(test_X)
        assert len(test_y) == self.test_num_trials, (
            'Different number of trials in test X and y')

        self.test_X, self.test_y, self.test_trials = self.iter_by_session(test_X, test_y)
        assert len(self.test_y) == len(self.test_X), (
            'Different number of sessions in test X and y')
        self.input_dim = self.train_X[0].shape[1]


class GLMHMM(ModelData):

    '''
    Fit and characterize GLM-HMM, using and guided by the ssm package from
    the Linderman lab: https://github.com/lindermanlab/ssm/
    '''

    def __init__(self,
                 num_states,
                 obs_dim: int = 1,
                 num_cat: int = 2,
                 obs_kwargs: dict = None,
                 observations: str = 'input_driven_obs',
                 transitions: str = 'standard',
                 trans_kwargs: dict = None):

        super().__init__()

        self.num_states = np.array(num_states)
        self.obs_dim = obs_dim  # number of observed dims, (e.g. reward, RT)

        self.observation_kwargs = obs_kwargs
        self.observations = observations
        self.transitions = transitions
        self.transition_kwargs = trans_kwargs

    def init_model(self):

        '''
        Initialize N instances of the model, for each of k number of states
        in self.num_states.
        '''
        self.model = {}
        for i, k in enumerate(self.num_states):
            self.model[i] = ssm.HMM(k,
                                    self.obs_dim,
                                    self.input_dim,
                                    observations=self.observations,
                                    observation_kwargs=self.observation_kwargs,
                                    transitions=self.transitions,
                                    transition_kwargs=self.transition_kwargs
                                    )

    def fit_cv(self, n_iters=200, pval=0.1, reps=3):

        '''Fit models with CV with *pval* heldout and return LLs.'''
        lls = []
        scores = {'train': [], 'test': []}
        for i in self.model:
            train_scores, test_scores = cross_val_scores(self.model[i],
                                                         self.train_y,
                                                         self.train_X,
                                                         heldout_frac=pval,
                                                         n_repeats=reps,
                                                         verbose=True)
            ll = self.model[i].fit(self.train_y, inputs=self.train_X,
                                   method="em", num_iters=n_iters,
                                   initialize=False)
            scores['train'].append(train_scores)
            scores['test'].append(test_scores)
            lls.append(ll)

        return lls, scores

    def compare_k_states(self, scores, datasets=['train', 'test']):

        '''
        Plot train and test scores for each model and display confidence
        intervals.
        '''
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        for key in datasets:
            plt.scatter(self.num_states, [np.mean(s) for s in scores[key]],
                        label=key)
            plt.errorbar(self.num_states, [np.mean(s) for s in scores[key]],
                         [calc_ci(s)[2] for s in scores[key]], alpha=0.3)

        plt.legend(bbox_to_anchor=(1, 1))
        ax.set(xlabel="Number of states", ylabel="Log Probability",
               title="Cross Validation Scores with 95% CIs")

    def compare_k_states_no_err(self, scores, ylab='', datasets=['train', 'test'], **kwargs):

        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)

        for key in datasets:
            plt.scatter(self.num_states, scores[key], label=key)

        plt.legend(bbox_to_anchor=(1, 1))
        ax.set(xlabel="Number of states", ylabel=ylab)
        if ylim := kwargs.get('ylim', False):
            plt.ylim(ylim)
        sns.despine()

    def calc_log_likelihood(self, verbose=False, normalize=False,
                            as_bits=False):

        assert sum((normalize, as_bits)) < 2, 'cannot normalize and compute bits together'
        denom_train = self.train_num_trials if normalize else 1
        denom_test = self.test_num_trials if normalize else 1
        # print(f'{denom_train =} \n{denom_test=}')
        LL = {'train': np.zeros(len(self.num_states)),
              'test': np.zeros(len(self.num_states))}
        for i, model in self.model.items():
            ll_train = (model.log_likelihood(self.train_y, inputs=self.train_X)
                        / denom_train)
            if as_bits:
                ll0 = (self.model[0].log_likelihood(self.train_y, inputs=self.train_X)
                        / denom_train)
                denom = np.log(2) * self.train_num_trials
                ll_train = (ll_train - ll0) / denom
            LL['train'][i] = ll_train

            ll_test = (model.log_likelihood(self.test_y, inputs=self.test_X)
                       / denom_test)

            if as_bits:
                ll0 = (self.model[0].log_likelihood(self.test_y, inputs=self.test_X)
                        / denom_test)
                denom = np.log(2) * self.test_num_trials
                ll_test = (ll_test - ll0) / denom
            LL['test'][i] = ll_test
            if verbose:
                print((f'Model with {i} states:'
                       f'\n{"":>5}{"train LL":<8} = {LL["train"][i]:.2f}'
                       f'\n{"":>5}{"test LL":<8} = {LL["test"][i]:.2f}'))

        return LL

    def calc_aic(self, scores):

        aic = {}
        if 'train' in scores:
            aic['train'] = ((-2 * scores['train'] / self.train_num_trials)
                            + (2 * np.array(self.num_states)))
        if 'test' in scores:
            aic['test'] = ((-2 * scores['test'] / self.test_num_trials)
                           + (2 * np.array(self.num_states)))
        return aic

    def predict_state(self):

        self.train_states = []
        self.test_states = []

        for _, model in self.model.items():
            self.train_states.append([model.expected_states(y, X)[0]
                                     for y, X in zip(self.train_y, self.train_X)])
            self.test_states.append([model.expected_states(y, X)[0]
                                     for y, X in zip(self.test_y, self.test_X)])

    def pred_occupancy(self):

        self.train_max_prob_state = []
        self.test_max_prob_state = []
        self.train_occupancy = []
        self.test_occupancy = []
        self.train_occupancy_rates = []
        self.test_occupancy_rates = []
        for i in self.model:
            state_max_posterior = [np.argmax(posterior, axis=1) for posterior in self.train_states[i]]

            state_occupancies = np.zeros((i+1, len(self.train_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.train_max_prob_state.append(state_max_posterior)
            self.train_occupancy.append([make_onehot_array(max_post) for max_post in state_max_posterior])
            self.train_occupancy_rates.append(state_occupancies)

            state_max_posterior = [np.argmax(posterior, axis=1) for posterior in self.test_states[i]]
            state_occupancies = np.zeros((i+1, len(self.test_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.test_max_prob_state.append(state_max_posterior)
            self.test_occupancy.append([make_onehot_array(max_post) for max_post in state_max_posterior])
            self.test_occupancy_rates.append(state_occupancies)

    def predict_choice(self, accuracy=True, verbose=False, policy='greedy'):

        self.pchoice = []
        acc = []
        for i, model in self.model.items():

            glm_weights = -model.observations.params
            permutation = np.argsort(glm_weights[:, 0, 0])

            pred_states = np.concatenate(self.test_states[i], axis=0)
            posterior_probs = pred_states[:, permutation]
            pright = [np.exp(model.observations.calculate_logits(input=X))
                      for X in self.test_X]

            # Multiply posterior_probs with pright and sum over latent axis.
            pright = np.concatenate(pright, axis=0)[:, :, 1]
            pright = np.sum(np.multiply(posterior_probs, pright), axis=1)

            # Get the predicted label for each time step.
            if policy == 'greedy':
                pred_choice = np.around(pright, decimals=0).astype('int')
            elif policy == 'prob_match':
                pred_choice = (np.random.random(size=len(pright)) < pright).astype('int')
            self.pchoice.append(pred_choice)
            if accuracy:
                pred_accuracy = np.mean(np.concatenate(self.test_y, axis=0)[:, 0] == pred_choice)

                if verbose:
                    print(f'Model with {i} state(s) has a test predictive'
                          f'accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc

    def plot_state_probs(self, model_idx, sess_idx: int = 0,
                         as_occupancy: bool = False,
                         fill_state: bool = True):

        if as_occupancy:
            samples = self.test_occupancy[model_idx][sess_idx]
        else:
            samples = self.test_states[model_idx][sess_idx]

        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(model_idx + 1):
            plt.plot(samples[:, i], label=i, alpha=0.8)

        if fill_state:
            state_preds = self.test_max_prob_state[model_idx][sess_idx]
            transitions = np.diff(state_preds)
            t_idx = np.insert(np.where(transitions != 0), 0, 0)
            t_idx = np.append(t_idx, len(state_preds) - 2)
            state_ids = state_preds[t_idx + 1]

            for t_start, t_stop, state in zip(t_idx[:-1], t_idx[1:], state_ids):
                ax.fill_betweenx(y=[0, 1], x1=t_start, x2=t_stop, alpha=0.3,
                                 color=sns.color_palette()[state])

        ax.set(xlabel='trial', ylabel='prob')
        plt.legend(bbox_to_anchor=(1, 1), title='latent state')
        sns.despine()

        return fig, ax
