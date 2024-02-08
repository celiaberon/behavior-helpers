import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bh.utils import calc_ci, calc_sem, make_onehot_array

def get_attribute(model, metric, dataset='test'):

    match metric:
        case 'LL':
            return model.LL[dataset]
        case 'LL_norm':
            return model.LL_norm[dataset]
        case 'bits':
            return model.bits[dataset]
        case 'AIC':
            return model.aic[dataset]
        case 'accuracy':
            return model.acc[dataset]
        case 'weights':
            return -model.observations.params.squeeze(1)
        case 'tmat':
            return np.exp(model.transitions.params)[0]


def get_multi_mice_metrics(models, metric, **kwargs):

    if isinstance(models, dict):
        models = models.values()

    n_mice = len(models)
    for i, model in enumerate(models):

        mouse_metric = get_attribute(model, metric, **kwargs)

        if i == 0:
            if len(mouse_metric.shape) == 1:
                mice_metrics = np.zeros((n_mice, len(mouse_metric)))
                dims = 2
            elif len(mouse_metric.shape) == 2:
                mice_metrics = np.zeros((n_mice, mouse_metric.shape[0], mouse_metric.shape[1]))
                dims = 3
        
        if dims == 2:
            mice_metrics[i, :] = mouse_metric
        elif dims == 3:
            mice_metrics[i, :, :] = mouse_metric

    return mice_metrics


def get_dict_item_by_idx(d, idx):

    for i, (k, v) in enumerate(d.items()):
        if i == idx:
            return k, v


def plot_glm_weights(model, num_states,
                     with_tmat=True):
    
    
    multi_mice = False
    if isinstance(model, dict):
        multi_mice = True
    
    if with_tmat:
        fig, axs = plt.subplots(ncols=2, nrows=num_states if multi_mice else 1, figsize=(9, 3 * (multi_mice * 2)), dpi=80, width_ratios=(1, 1))
        plot_tmat(model, num_states, ax=axs[0, 1] if multi_mice else axs[1])
        if multi_mice:
            ax = axs[:,0].flatten() 
            [ax_.axis('off') for ax_ in axs[1:, 1]]
        else:
            axs[0]
    else:
        fig, ax = plt.subplots(nrows=num_states if multi_mice else 1,
                               figsize=(7, 2.5*(multi_mice*2)), dpi=80)

    if multi_mice:
        models = []
        for _, m_ in model.items():
            model_idx = np.where(m_.num_states == num_states)[0][0]
            models.append(m_.model[model_idx])
        weights = get_multi_mice_metrics(models, 'weights')
        model = m_ # for convenience
    else:
        model_idx = np.where(model.num_states == num_states)[0][0]
        weights = -model.model[model_idx].observations.params

    c = {i: sns.color_palette()[i] for i in range(num_states)}
    for k in range(num_states):
        ax_ = ax[k] if multi_mice else ax
        for grp in np.arange(len(model.features), step=model.nlags):
            
            if multi_mice:
                [ax_.plot(range(grp, grp+model.nlags),
                     weights[imouse, k, grp:grp+model.nlags],
                     label=None, 
                     lw=1, color=c.get(k)) for imouse in range(weights.shape[0])]
                ax_.plot(range(grp, grp+model.nlags),
                     np.mean(weights, axis=0)[k, grp:grp+model.nlags],
                     label=None, 
                     lw=2, marker='o', markersize=5, color='k')

            else: 
                ax_.plot(range(grp, grp+model.nlags),
                     weights[0, k, grp:grp+model.nlags],
                     label=f'State {k + 1}' if grp==0 else None, 
                     lw=2, marker='o', markersize=5, color=c.get(k))

        ax_.hlines(xmin=-1, xmax=len(model.features) + 1, y=0, color='k', lw=0.5)
        ax_.set(xlabel='Features', xlim=(0, len(model.features)),
                ylabel='GLM weight', )
        ax_.set_xticks(np.arange(len(model.features)), model.features,
                    rotation=45, fontsize=10)

    if not multi_mice:
        ax_.legend(bbox_to_anchor=(1, 1), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    sns.despine()


def plot_tmat(model, num_states, ax=None):

    if isinstance(model, dict):
        models = []
        for _, m_ in model.items():
            model_idx = np.where(m_.num_states == num_states)[0][0]
            models.append(m_.model[model_idx])
        tmat = np.mean(get_multi_mice_metrics(models, 'tmat'), axis=0)
    else:
        model_idx = np.where(model.num_states == num_states)[0][0]
        tmat = np.exp(model.model[model_idx].transitions.params)[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
    ax.imshow(tmat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(tmat.shape[0]):
        for j in range(tmat.shape[1]):
            _ = ax.text(j, i, str(np.around(tmat[i, j], decimals=2)),
                            ha="center", va="center", color="k", fontsize=10)
    ax.set(xlabel='state t+1', xlim=(-0.5, num_states - 0.5),
            ylabel='state t', ylim=(num_states - 0.5, -0.5),
            title='Transition matrix')
    ax.set_xticks(np.arange(num_states), range(1, num_states+1), fontsize=10)
    ax.set_yticks(np.arange(num_states), range(1, num_states+1), fontsize=10)
    plt.tight_layout()


def compare_k_states(models, metric, datasets=['train', 'test'], **kwargs):

    # plot train and test scores for each model and display confidence intervals

    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    for key in datasets:
        if isinstance(models, dict):
            scores = get_multi_mice_metrics(models, metric, dataset=key)
            num_states = get_dict_item_by_idx(models, 0)[1].num_states
        else:
            scores = get_attribute(models, metric, dataset=key)
            num_states = models.num_states
        plt.scatter(num_states, np.mean(scores, axis=0),
                    label=key)
        plt.errorbar(num_states, np.mean(scores, axis=0),
                        [calc_ci(s)[2] for s in scores.T], alpha=0.3)

    plt.legend(bbox_to_anchor=(1, 1))
    ax.set(xlabel="Number of states", xticks=num_states, **kwargs)
    sns.despine()


def compare_k_states_no_err(model, metric, datasets=['train', 'test'], **kwargs):

    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    for key in datasets:
        scores = get_attribute(model, metric, dataset=key)
        plt.scatter(model.num_states, scores, label=key)

    plt.legend(bbox_to_anchor=(1, 1))
    ax.set(xlabel="Number of states", **kwargs)
    sns.despine()