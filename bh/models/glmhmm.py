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
                 obs_kwargs: dict = None,
                 transitions: str = 'standard',
                 trans_kwargs: dict = None):

        '''
        Initialize class with parameters for GLM-HMM. Below arguments are
        parameters and definitions as found in the ssm package linked above.

        Args:
            num_states:
                List of number of states to fit model with. Results in
                len(num_states) models being fit.
            obs_dim:
                Dimensionality of the observations (e.g., 1 when only
                predicting choice, 2 when predicting choice AND reaction time).
            obs_kwargs:
                'C': number of classes (choices) for output. Always 2 for 2ABT.
                'prior_sigma': Strength of prior, defaults to 1000, where MAP
                converges to MLE. Smaller sigma may be better in low data
                regime.
                'prior_mean': Defaults to 0.
            transitions:
                Defaults to 'standard', which is equivalent to 'stationary'.
                Might also consider 'sticky'.
            trans_kwargs:
                If transitions is 'sticky', additional kwargs:
                    'alpha': Defaults to 1.
                    'kappa': Defaults to 100.
                    Weights as Dir(alpha + kappa * e_k) to strengthen prior
                    on current state k.
        '''

        super().__init__()

        # Observations will always be input driven observations, which allows
        # for learning GLM weights on data.
        self.observations = 'input_driven_obs'

        self.num_states = np.array(num_states)
        self.obs_dim = obs_dim  # number of observed dims, (e.g. reward, RT)

        # Keyword arguments passed to ssm.HMM()
        self.observation_kwargs = obs_kwargs
        self.transitions = transitions  # 'standard' or 'sticky'
        self.transition_kwargs = trans_kwargs

    def init_model(self):

        '''
        Initialize an instance of the model for each of k number of states
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

        '''
        Fit models with CV with fraction of data (pval) held out and return LLs.
        '''
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

    def compare_k_states(self,
                         scores,
                         datasets: list[str] = ['train', 'test']):

        '''
        Plot train and test scores for each model and display confidence
        intervals.
        '''
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80, layout='constrained')
        for key in datasets:
            plt.scatter(self.num_states, [np.mean(s) for s in scores[key]],
                        label=key)
            plt.errorbar(self.num_states, [np.mean(s) for s in scores[key]],
                         [calc_ci(s)[2] for s in scores[key]], alpha=0.3)
        plt.legend(bbox_to_anchor=(1, 1))
        ax.set(xlabel='Number of states', ylabel='Log Probability',
               title='CV Scores with 95% CIs')
        sns.despine()

    def compare_k_states_no_err(self,
                                scores,
                                ylab: str = '',
                                datasets: list[str] = ['train', 'test'],
                                **kwargs):
        '''Plot train and test scores for each model without error bars.'''
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80, layout='constrained')

        for key in datasets:
            plt.scatter(self.num_states, scores[key], label=key)
        plt.legend(bbox_to_anchor=(1, 1))
        ax.set(xlabel='Number of states', ylabel=ylab)
        if ylim := kwargs.get('ylim', False):
            plt.ylim(ylim)
        sns.despine()

    def calc_log_likelihood(self,
                            verbose: bool = False,
                            normalize: bool = False,
                            as_bits: bool = False) -> dict[str, np.array]:

        '''
        Calclulate log likelihoood for train and test sets.
        Args:
            verbose:
                If True, print LL scores.
            normalize:
                If True, normalize LL by number of trials in each dataset as
                (LL / num_trials).
            as_bits:
                If True, compute LL in units of bits as
                (LL - LL0) / (np.log(2) * num_trials)
                where LL0 is the LL for a standard GLM (1 state model).
                Note: mutually exclusive with normalize.
        '''
        assert sum((normalize, as_bits)) < 2, (
            'Cannot normalize and compute bits together')

        if as_bits:
            assert self.num_states[0] == 1, (
                'Bitwise LL needs 1 state model to reference for LL0.'
            )

        # Divide by number of trials if normalizing likelihood.
        denom_train = self.train_num_trials if normalize else 1
        denom_test = self.test_num_trials if normalize else 1

        LL = {'train': np.zeros(len(self.num_states)),
              'test': np.zeros(len(self.num_states))}
        for i, model in self.model.items():

            # Train LL.
            ll_train = (model.log_likelihood(self.train_y, inputs=self.train_X)
                        / denom_train)
            if as_bits:
                ll0 = (self.model[0].log_likelihood(self.train_y, inputs=self.train_X)
                       / denom_train)
                denom = np.log(2) * self.train_num_trials
                ll_train = (ll_train - ll0) / denom
            LL['train'][i] = ll_train

            # Test LL.
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

        '''
        Calculate Aikake Information Criterion as AIC = -2*LL/N + 2K,
        where LL/N is the average LL per trial, and K is the number of
        parameters in the model (penalty).

        Because each state contains the same number of weights, we factor that
        out to penalize simply by number of states.
        '''
        aic = {}
        if 'train' in scores:
            aic['train'] = ((-2 * scores['train'] / self.train_num_trials)
                            + (2 * np.array(self.num_states)))
        if 'test' in scores:
            aic['test'] = ((-2 * scores['test'] / self.test_num_trials)
                           + (2 * np.array(self.num_states)))
        return aic

    def predict_state(self):

        '''
        Store state probability distribution for each trial (with same outer
        dims as train/test sets) using function ssm.hmm.expected_states().
        '''

        self.train_states = []
        self.test_states = []

        for _, model in self.model.items():
            self.train_states.append([model.expected_states(y, X)[0]
                                     for y, X in zip(self.train_y, self.train_X)])
            self.test_states.append([model.expected_states(y, X)[0]
                                     for y, X in zip(self.test_y, self.test_X)])

    def pred_occupancy(self):

        '''
        Predict most likely state for each trial (as one hot encoding), and
        calculate state occupancies as average number of trials predicted in
        each of num_states per session.
        '''

        self.train_max_prob_state = []
        self.test_max_prob_state = []
        self.train_occupancy = []
        self.test_occupancy = []
        self.train_occupancy_rates = []
        self.test_occupancy_rates = []
        for i in self.model:
            # List of which state was most likely for each trial in each session.
            state_max_posterior = [np.argmax(posterior, axis=1)
                                   for posterior in self.train_states[i]]
            self.train_max_prob_state.append(state_max_posterior)

            # Count number of trials in each state for each session.
            state_occupancies = np.zeros((i+1, len(self.train_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            # Calculate average occupancy in each state over sessions.
            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.train_occupancy_rates.append(state_occupancies)

            # Make one hot array predicting which state each trial is in, greedy.
            self.train_occupancy.append([make_onehot_array(max_post)
                                         for max_post in state_max_posterior])

            # Same as above for test set.
            state_max_posterior = [np.argmax(posterior, axis=1)
                                   for posterior in self.test_states[i]]
            self.test_max_prob_state.append(state_max_posterior)

            state_occupancies = np.zeros((i+1, len(self.test_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.test_occupancy_rates.append(state_occupancies)

            self.test_occupancy.append([make_onehot_array(max_post)
                                        for max_post in state_max_posterior])

    def predict_choice(self, accuracy=True, verbose=False, policy='greedy'):

        '''
        Predict choice probability based on GLM logit and underlying state
        probabilities from HMM. Convert to binary prediction according to
        policy, and calculate prediction accuracy if requested. For test set
        only.
        '''
        self.pchoice = []
        acc = []
        for i, model in self.model.items():

            # glm_weights = -model.observations.params  # why negative?
            # Permutation just sorts by weight on first feature...?
            # permutation = np.argsort(glm_weights[:, 0, 0])

            # Instead, shouldn't states already be in same order as their
            # weights?
            permutation = [i for i in range(self.num_states[i])]

            # Take state prob distributions and permute based on above.
            pred_states = np.concatenate(self.test_states[i], axis=0)
            posterior_probs = pred_states[:, permutation]

            # Convert state probs to choice prob using GLM.
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
                    print(f'Model with {i} state(s) has a test predictive '
                          f'accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc

    def plot_state_probs(self,
                         model_idx,
                         sess_idx: int = 0,
                         as_occupancy: bool = False,
                         fill_state: bool = True):

        '''
        Plot state predictions for a given session.
        Args:
            model_idx:
                Index of model for list of models defined by self.num_states.
            sess_idx:
                Index of session to plot.
            as_occupancy:
                If True, plot predicted state; if false, plot all state
                probabilities per trial.
            fill_state:
                If True, shade trials by state occupancy.
        '''

        num_states = self.num_states[model_idx]
        if as_occupancy:
            samples = self.test_occupancy[model_idx][sess_idx]
        else:
            samples = self.test_states[model_idx][sess_idx]

        fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

        # Plot trace (probability or prediction) for each possible state.
        for i in range(num_states + 1):
            plt.plot(samples[:, i], label=i, alpha=0.8)

        # Shade predicted states between state transitions.
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
