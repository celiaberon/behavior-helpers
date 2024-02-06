import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ssm
from sklearn.model_selection import train_test_split
from ssm.model_selection import cross_val_scores

sys.path.append(f'{os.path.expanduser("~")}/GitHub/neural-timeseries-analysis/')
from nta.features import behavior_features as bf

from bh.utils import calc_ci, calc_sem


class GLMHMM:

    def __init__(self,
                 num_states,
                 obs_dim: int = 1,
                 num_cat: int = 2,
                 C: int = 2,
                 prior_sigma: int = None,
                 prior_alpha: int = None):
        
        self.num_states = np.array(num_states)
        self.obs_dim = obs_dim  # number of observed dimensions, 1 for just reward, 2 if you had something like reaction time
        self.num_cat = num_cat  # number of categories for the output
        self.C = C  # number of classes for prediction?

        # Priors on the weights.
        self.prior_sigma = prior_sigma
        self.prior_alpha = prior_alpha

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

        self.y = trials_clean[[yvar, 'Session']].copy()
        self.X = trials_clean[initial_cols + ['Session']].copy()

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
        self.train_X = (self.X.query('Session.isin(@train_ids)')
                        .drop(columns=['Session'])
                        .to_numpy())
        self.train_y = (self.y.query('Session.isin(@train_ids)')
                        .drop(columns=['Session'])
                        .to_numpy()
                        .reshape(-1, 1)
                        .astype(int))
        self.test_X = (self.X.query('Session.isin(@test_ids)')
                       .drop(columns=['Session'])
                       .to_numpy())
        self.test_y = (self.y.query('Session.isin(@test_ids)')
                       .drop(columns=['Session'])
                       .to_numpy()
                       .reshape(-1, 1)
                       .astype(int))
        self.train_num_trials = len(self.train_X)
        self.test_num_trials = len(self.test_X)

        self.input_dim = self.train_X.shape[1]

    def init_model(self, **kwargs):

        self.model = {}
        for i, k in enumerate(self.num_states):
            # Initialize model
            self.model[i] = ssm.HMM(k,
                                    self.obs_dim,
                                    self.input_dim,
                                    observations="input_driven_obs",
                                    observation_kwargs={
                                        'C': self.num_cat,
                                        'prior_sigma': self.prior_sigma},
                                    **kwargs
                                )

    def fit_cv(self, n_iters=200, pval=0.1, reps=3):

        lls = []
        scores = {'train': [], 'test': []}
        for i in self.model:

            train_scores, test_scores = cross_val_scores(self.model[i],
                                                         self.train_y,
                                                         self.train_X,
                                                         heldout_frac=pval, n_repeats=reps, verbose=True)
            ll = self.model[i].fit(self.train_y, inputs=self.train_X, method="em", num_iters=n_iters, initialize=False)
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

    def compare_k_states_no_err(self, scores, ylab='', datasets=['train', 'test']):

        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)

        for key in datasets:
            plt.scatter(self.num_states, scores[key], label=key)

        plt.legend(bbox_to_anchor=(1, 1))
        ax.set(xlabel="Number of states", ylabel=ylab)
        sns.despine()

    def calc_log_likelihood(self, verbose=False, normalize=False, as_bits=False):


        assert sum((normalize, as_bits)) < 2, 'cannot normalize and compute bits together'
        denom_train = self.train_num_trials if normalize else 1
        denom_test = self.test_num_trials if normalize else 1

        LL = {'train': np.zeros(len(self.num_states)),
              'test': np.zeros(len(self.num_states))}
        for i, model in self.model.items():
            ll_train = (model.log_likelihood(self.train_y, inputs=self.train_X)
                        / denom_train)
            if as_bits:
                ll0 = np.log(0.5) * self.train_num_trials
                denom = np.log(2) * self.train_num_trials
                ll_train = (ll_train - ll0) / denom
            LL['train'][i] = ll_train

            ll_test = (model.log_likelihood(self.test_y, inputs=self.test_X)
                       / denom_test)

            if as_bits:
                ll0 = np.log(0.5) * self.test_num_trials
                denom = np.log(2) * self.test_num_trials
                ll_test = (ll_test - ll0) / denom
            LL['test'][i] = ll_test
            if verbose:
                print((f'Model with {i} states:'
                       f'\n{"":>5}{"train LL":<8} = {LL["train"][-1]:.2f}'
                       f'\n{"":>5}{"test LL":<8} = {LL["test"][-1]:.2f}'))

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
            self.train_states.append(model.expected_states(self.train_y, self.train_X)[0])
            self.test_states.append(model.expected_states(self.test_y, self.test_X)[0])

    def predict_choice(self, accuracy=True, verbose=False):

        self.pchoice = []
        acc = []
        for i, model in self.model.items():

            glm_weights = -model.observations.params
            permutation = np.argsort(glm_weights[:, 0, 0])

            pred_states = self.test_states[i]
            posterior_probs = pred_states[:, permutation]
            pright = np.exp(model.observations.calculate_logits(input=self.test_X))[:, :, 1]

            # Now multiply posterior probs and prob_right and sum over latent axis.
            pright = np.sum(np.multiply(posterior_probs, pright), axis=1)

            # Get the predicted label for each time step.
            pred_choice = np.around(pright, decimals=0).astype('int')
            self.pchoice.append(pred_choice)
            if accuracy:
                pred_accuracy = np.mean(self.test_y[:, 0] == pred_choice)

                if verbose:
                    print(f'Model with {i} state(s) has a test predictive accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc

    def plot_state_probs(self, model_idx, num_trials: int = None):

        if num_trials is None:
            num_trials = len(self.train_states[model_idx])

        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(model_idx+1):
            plt.plot(self.train_states[model_idx][:, i][:num_trials], label=i)
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
