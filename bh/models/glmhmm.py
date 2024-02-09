import os
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ssm
from sklearn.model_selection import train_test_split
from ssm.model_selection import cross_val_scores

sys.path.append(f'{os.path.expanduser("~")}/GitHub/neural-timeseries-analysis/')
from nta.features import behavior_features as bf

from bh.utils import calc_ci, calc_sem, make_onehot_array


class ModelData(ABC):

    def __init__(self):
        pass

    def drop_nans(self):
        pass

    def prepare_features(self, trials, yvar, feat_funcs=None, nlags=3):

        self.nlags = nlags

        def pm1(x):
            return 2 * x - 1
        if feat_funcs is None:
            feat_funcs = {'Reward': lambda r, c: r,
                          'direction': lambda r, c: pm1(c),
                          'dir-rew': lambda r, c: r * pm1(c)
                          }

        initial_cols = [col for col in feat_funcs if col in trials.columns]
        trials_clean = trials.dropna(subset=['Reward', 'direction'])

        self.y = trials_clean[[yvar, 'Session', 'nTrial']].copy()
        self.X = trials_clean[initial_cols + ['Session', 'nTrial']].copy()

        # Forward and backward shifts that can be useful (need to shift up front).
        for feature, func in feat_funcs.items():

            self.X[feature] = func(trials_clean['Reward'].values, trials_clean['direction'].values)
            print(self.X[feature].unique())
            for lag in range(1, nlags + 1):
                self.X = bf.shift_trial_feature(self.X, col=feature,
                                                n_shift=lag,
                                                shift_forward=True)

        assert all([col in self.X.columns for col in list(feat_funcs.keys())])

        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        lag_nulls = np.unique(np.where(np.isnan(self.X.drop(columns=['Session'])))[0])
        self.X = self.X.drop(index=lag_nulls)
        self.y = self.y.drop(index=lag_nulls)
        self.X = self.X.drop(columns=list(feat_funcs.keys()))
        self.features = self.X.columns.drop('Session').values

    def get_data_subset(self, dataset='X', col='Session', vals=None):

        if dataset == 'X':
            return self.X.copy().query(f'{col}.isin(@vals)').reset_index(drop=True)

        else:
            return self.y.copy().query(f'{col}.isin(@vals)').reset_index(drop=True)

    def iter_by_session(self, X, y):

        session_starts = X.groupby('Session').nth(0).index.values
        session_starts = np.concatenate((session_starts, [len(X)]))
        trial_ids = [X.loc[start:stop].nTrial.values
                     for start, stop in zip(session_starts, session_starts[1:])]
        X = X.drop(columns=['Session', 'nTrial'])
        y = y.drop(columns=['Session', 'nTrial'])

        X_by_sess = [X.loc[start:stop].to_numpy()
                     for start, stop in zip(session_starts, session_starts[1:])]
        y_by_sess = [y.loc[start:stop].to_numpy().reshape(-1, 1).astype('int')
                     for start, stop in zip(session_starts, session_starts[1:])]

        return X_by_sess, y_by_sess, trial_ids

    def split_data(self, ptrain=0.8, seed=0, verbose=False):

        # Assign session ids to train and test splits
        train_ids, test_ids = train_test_split(self.X['Session'].unique(),
                                               train_size=ptrain,
                                               random_state=seed)

        if verbose:
            print(f'{len(train_ids)} training sessions and {len(test_ids)} test sessions')

        assert len(train_ids) == len(set(train_ids))
        assert len(test_ids) == len(set(test_ids))

        self.train_num_sess = len(train_ids)
        self.test_num_sess = len(test_ids)

        # Split data into train and test sets
        train_X = self.get_data_subset(dataset='X', col='Session',
                                       vals=train_ids)
        train_y = self.get_data_subset(dataset='y', col='Session',
                                       vals=train_ids)
        self.train_num_trials = len(train_X)
        assert len(train_y) == self.train_num_trials, (
            'Different number of trials in train X and y')

        self.train_X, self.train_y, self.train_trials = self.iter_by_session(train_X, train_y)
        assert len(self.train_y) == len(self.train_X), (
            'Different number of sessions in train X and y')

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

    def __init__(self,
                 num_states,
                 obs_dim: int = 1,
                 num_cat: int = 2,
                #  C: int = 2,
                 obs_kwargs: dict = None,
                 observations: str = 'input_driven_obs',
                 transitions: str = 'standard',
                 trans_kwargs: dict = None):

        super().__init__()

        self.num_states = np.array(num_states)
        self.obs_dim = obs_dim  # number of observed dimensions, (e.g. reward, reaction time)

        self.observation_kwargs = obs_kwargs
        self.observations = observations
        self.transitions = transitions
        self.transition_kwargs = trans_kwargs

    def init_model(self):

        self.model = {}
        for i, k in enumerate(self.num_states):
            # Initialize model
            self.model[i] = ssm.HMM(k,
                                    self.obs_dim,
                                    self.input_dim,
                                    observations=self.observations,
                                    observation_kwargs=self.observation_kwargs,
                                    transitions=self.transitions,
                                    transition_kwargs=self.transition_kwargs
                                )

    def fit_cv(self, n_iters=200, pval=0.1, reps=3):

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

        # plot train and test scores for each model and display confidence intervals
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
        # self.pchoice = []

        for i, model in self.model.items():
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

    def predict_choice(self, accuracy=True, verbose=False):

        self.pchoice = []
        acc = []
        for i, model in self.model.items():

            glm_weights = -model.observations.params
            permutation = np.argsort(glm_weights[:, 0, 0])

            pred_states = np.concatenate(self.test_states[i], axis=0)
            posterior_probs = pred_states[:, permutation]
            pright = [np.exp(model.observations.calculate_logits(input=X))
                      for X in self.test_X]

            # Now multiply posterior probs and prob_right and sum over latent axis.
            pright = np.concatenate(pright, axis=0)[:, :, 1]
            pright = np.sum(np.multiply(posterior_probs, pright), axis=1)

            # Get the predicted label for each time step.
            pred_choice = np.around(pright, decimals=0).astype('int')
            self.pchoice.append(pred_choice)
            if accuracy:
                pred_accuracy = np.mean(np.concatenate(self.test_y, axis=0)[:, 0] == pred_choice)

                if verbose:
                    print(f'Model with {i} state(s) has a test predictive accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc


    def plot_state_probs(self, model_idx, sess_idx: int = 0,
                         as_occupancy: bool = False):

        if as_occupancy:
            samples = self.test_occupancy[model_idx][sess_idx]
        else:
            samples = self.test_states[model_idx][sess_idx]

        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(model_idx + 1):
            plt.plot(samples[:, i], label=i, alpha=0.8)
        ax.set(xlabel='trial', ylabel='prob')
        plt.legend(bbox_to_anchor=(1, 1), title='latent state')
        sns.despine()

    def plot_glm_weights(self, model_idx, with_tmat=True):

        weights = -self.model[model_idx].observations.params

        c = {i: sns.color_palette()[i] for i in range(len(self.num_states))}
        if with_tmat:
            fig, axs = plt.subplots(ncols=2, figsize=(9, 3), dpi=80, width_ratios=(1, 1))
            self.plot_tmat(model_idx, ax=axs[1])
            ax = axs[0]
        else:
            fig, ax = plt.subplots(figsize=(7, 2.5), dpi=80)

        for k in range(model_idx + 1):
            for grp in np.arange(len(self.features), step=self.nlags):
                ax.plot(range(grp, grp+self.nlags),
                         weights[k, 0, grp:grp+self.nlags],
                         label=f'State {k + 1}' if grp==0 else None, 
                         lw=2, marker='o', markersize=5, color=c.get(k))

        ax.hlines(xmin=-1, xmax=len(self.features) + 1, y=0, color='k', lw=1)
        ax.legend(bbox_to_anchor=(1, 1), frameon=False)
        ax.set(xlabel='Features', xlim=(0, len(self.features)),
               ylabel='GLM weight', )
        ax.set_xticks(np.arange(len(self.features)), self.features,
                   rotation=45, fontsize=10)

        plt.tight_layout()
        sns.despine()

    def plot_tmat(self, model_idx, ax=None):

        tmat = np.exp(self.model[model_idx].transitions.params)[0]
        plot_states = self.num_states[:model_idx+1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
        ax.imshow(tmat, vmin=-0.8, vmax=1, cmap='bone')
        for i in range(tmat.shape[0]):
            for j in range(tmat.shape[1]):
                _ = plt.text(j, i, str(np.around(tmat[i, j], decimals=2)),
                             ha="center", va="center", color="k", fontsize=10)
        ax.set(xlabel='state t+1', xlim=(-0.5, len(plot_states) - 0.5),
               ylabel='state t', ylim=(len(plot_states) - 0.5, -0.5),
               title='Transition matrix')
        plt.xticks(plot_states - 1, plot_states, fontsize=10)
        plt.yticks(plot_states - 1, plot_states, fontsize=10)
        plt.tight_layout()

