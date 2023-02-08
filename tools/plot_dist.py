import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as skm
import scipy.stats as stats
from pomegranate import GeneralMixtureModel, TrueBetaDistribution
import argparse

def plot_data(data, ax, bins=np.linspace(0., 1., 100), name=None):
    ax.hist(data, bins, alpha=0.5, density=True, label=name)

def plot_density(bins, pdf, ax, name=None):
    assert len(bins) == len(pdf)
    ax.plot(bins, pdf, label=name)

def bmm_policy(scores):
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
    min_score = scores.min()
    max_score = scores.max()
    mean1 = min_score + 0.1 * (max_score - min_score)
    mean2 = min_score + 0.9 * (max_score - min_score)
    d1 = TrueBetaDistribution(10 * mean1, 10 * (1 - mean1))
    d2 = TrueBetaDistribution(10 * mean2, 10 * (1 - mean2))
    model = GeneralMixtureModel([d1, d2])
    model.fit(scores.reshape(-1, 1))
    a1, b1 = model.distributions[0].parameters
    a2, b2 = model.distributions[1].parameters
    w1, w2 = model.to_dict()['weights']
    p1 = [a1, b1, w1]
    p2 = [a2, b2, w2]
    mean1 = a1 / (a1 + b1)
    mean2 = a2 / (a2 + b2)
    if mean1 > mean2:
        p1, p2 = p2, p1
    params = [p1, p2]
    return np.array(params)

def gmm_policy(scores, given_gt_thr=0.9):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 2:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        means = gmm.means_
        weights = gmm.weights_
        vars = np.sqrt(gmm.covariances_[..., 0])
        params = np.concatenate([means, vars, weights.reshape(-1, 1)], 1)
        return params, gmm.predict_proba(scores)

def get_bmm_info(scores, params):
    scores_backup = scores
    if isinstance(scores_backup, torch.Tensor):
        scores = scores_backup.cpu().numpy()
        params = params.cpu().numpy()
    a1, b1, w1 = params[0]
    a2, b2, w2 = params[1]
    d1 = TrueBetaDistribution(a1, b1)
    d2 = TrueBetaDistribution(a2, b2)
    if len(scores) == 1:
        pdf1 = d1.probability(scores.repeat(2))[0:1]
        pdf2 = d2.probability(scores.repeat(2))[0:1]
    else:
        pdf1 = d1.probability(scores)
        pdf2 = d2.probability(scores)
    is_pos = pdf2 > pdf1
    if is_pos.sum() == 0:
        thr = float(scores.max()) + 1e-8
    else:
        thr = float(scores[is_pos].min()) - 1e-8
    weight = w2 * pdf2 / (w1 * pdf1 + w2 * pdf2 + 1e-8)
    weight[~is_pos] = 1
    # print(scores[is_pos], weight[is_pos], flush=True)
    if isinstance(scores_backup, torch.Tensor):
        is_pos = torch.from_numpy(is_pos).to(scores_backup)
        weight = torch.from_numpy(weight).to(scores_backup)
    return is_pos, weight, thr

def min_normalize(value, thr=0.):
    return (value - thr) / (1 - thr)

def min_denormalize(value, thr=0.):
    return value * (1 - thr) + thr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--method", default='gmm', help="gmm or bmm")
    parser.add_argument("--normalize", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pth = torch.load(args.ckpt, map_location='cpu')
    scores_orig = pth['state_dict']['scores'].sort(-1)[0].numpy()
    # params_all = pth['state_dict']['gmm_params'].numpy()
    if args.normalize:
        scores = min_normalize(scores_orig)
    else:
        scores = scores_orig
    num_classes = scores.shape[0]
    classes = pth['meta']['CLASSES']
    fig, axes = plt.subplots(num_classes, figsize=(10, 4 * num_classes))
    thresholds = np.zeros(num_classes)
    for i in range(num_classes):
        if args.normalize:
            scores_class = scores_orig[i]
        else:
            scores_class = scores[i]
        if args.method == 'gmm':
            params, prob = gmm_policy(scores[i], given_gt_thr=0.9)
            axes[i].plot(scores[i], stats.norm.pdf(scores[i], params[0, 0], params[0, 1]))
            axes[i].plot(scores[i], stats.norm.pdf(scores[i], params[1, 0], params[1, 1]))
        elif args.method == 'bmm':
            params = bmm_policy(scores[i])
            _, weights_, thr = get_bmm_info(scores[i], params)
            if args.normalize:
                thr = min_denormalize(thr)
            axes[i].plot(scores[i],
                         stats.beta.pdf(scores[i], params[0, 0], params[0, 1]))
            axes[i].plot(scores[i],
                         stats.beta.pdf(scores[i], params[1, 0], params[1, 1]))
            axes[i].vlines(thr, 0, 5)
            thresholds[i] = thr
        plot_data(scores_class, axes[i])
        axes[i].set_title(f'{classes[i]}')
    print(thresholds.mean())
    fig.tight_layout()
    fig.savefig('dist.pdf')
