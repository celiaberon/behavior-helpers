import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bh.utils import calc_ci, calc_sem, make_onehot_array, get_dict_item_by_idx
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

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


def plot_glm_weights(model, num_states,
                     with_tmat=True):
    
    from matplotlib.gridspec import GridSpec
    multi_mice = False
    if isinstance(model, dict):
        multi_mice = True
        n_rows = num_states
        state_iter = range(num_states)
    else:
        n_rows = 1
        state_iter = range(1, num_states+1) if isinstance(num_states, int) else num_states
    
    if with_tmat:
        fig = plt.figure(layout='constrained', figsize=(7, (1.5 * n_rows)+0.5))
        gs = GridSpec(ncols=3, nrows=n_rows, figure=fig)
        weight_ax = [fig.add_subplot(gs[i, :2]) for i in range(n_rows)]
        print(len(weight_ax))
        [weight_ax[i].sharey(weight_ax[i+1]) for i in range(len(weight_ax)-1)] 
        if multi_mice:
            tmat_ax = fig.add_subplot(gs[:, -1])
        else:
            tmat_ax = fig.add_subplot(gs[0, -1])
        ax = weight_ax
        plot_tmat(model, num_states, ax=tmat_ax)

    else:
        fig, ax = plt.subplots(nrows=num_states if multi_mice else 1,
                               figsize=(7, 1.5*(multi_mice*1 + 1)), dpi=80, sharex=True, sharey=True)

    if multi_mice:
        models = []
        for _, m_ in model.items():
            model_idx = np.where(m_.num_states == num_states)[0][0]
            models.append(m_.model[model_idx])
        # weight dims: (imouse, latent state, num_features)
        weights = get_multi_mice_metrics(models, 'weights')
        model = m_ # for convenience

    c = {i: sns.color_palette()[i] for i in state_iter}
    for i, k in enumerate(state_iter):
        ax_ = ax[i] if n_rows>1 else ax[0]
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
                model_idx = np.where(model.num_states == k)[0][0]
                 # weight dims: (latent state, always zero, num_features)
                weights = -model.model[model_idx].observations.params
                ax_.plot(range(grp, grp+model.nlags),
                     weights[i, 0, grp:grp+model.nlags],
                     label=f'State {k}' if grp==0 else None, 
                     lw=2, marker='o', markersize=5, color=c.get(k))

        ax_.hlines(xmin=-1, xmax=len(model.features) + 1, y=0, color='k', lw=0.5)
        ax_.set(xlabel='', xlim=(0, len(model.features)),
                ylabel='GLM weight', )
        ax_.set_xticks([])
    
    ax_.set(xlabel='Features')
    ax_.set_xticks(np.arange(len(model.features)), model.features,
                rotation=45, fontsize=10)

    if not multi_mice:
        ax_.legend(bbox_to_anchor=(1.05,1), frameon=False)

    sns.despine()
    fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
                            wspace=0)


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
            # title='Transition matrix'
            )
    ax.set_xticks(np.arange(num_states), range(1, num_states+1), fontsize=10)
    ax.set_yticks(np.arange(num_states), range(1, num_states+1), fontsize=10)
    # plt.tight_layout()


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


def add_sunburst_subplot(occ_counts, mouse, fig, j, ring_range=(0,4), **kwargs):

    fig_tmp = px.sunburst(occ_counts,
                          path=[f'model_state{i}' for i in range(*ring_range)],
                          values='count',
                          title=mouse
                          )
    fig_tmp.update(layout_coloraxis_showscale=False)

    n_rows = fig._get_subplot_rows_columns()[0][-1]
    fig.add_trace(go.Sunburst(
                  labels=fig_tmp['data'][0]['labels'].tolist(),
                  parents=fig_tmp['data'][0]['parents'].tolist(),
                  values=fig_tmp['data'][0]['values'].tolist(),
                  ids=fig_tmp['data'][0]['ids'].tolist(),
                  **kwargs
                            ),
                 row=int(j//n_rows)+1, col=int(j%n_rows)+1
                )
    return fig