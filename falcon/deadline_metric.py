"""Training-only KISSME covariance and PSD metric core."""
import numpy as np


def pair_covariances(features, labels):
    """Exact ordered distinct pair second moments, without forming N x N x D."""
    x = np.asarray(features, np.float64)
    labels = np.asarray(labels)
    if len(x) != len(labels) or len(np.unique(labels)) < 2:
        raise ValueError('Need aligned rows and at least two identities')
    centered = x - x.mean(0)
    total = 2 * len(x) * (centered.T @ centered)
    same_sum = np.zeros_like(total)
    same_count = 0
    for label in np.unique(labels):
        group = x[labels == label]
        residual = group - group.mean(0)
        same_sum += 2 * len(group) * (residual.T @ residual)
        same_count += len(group) * (len(group) - 1)
    different_count = len(x) * (len(x) - 1) - same_count
    if same_count == 0 or different_count == 0:
        raise ValueError('Need both same-ID and different-ID pairs')
    return same_sum / same_count, (total - same_sum) / different_count


def psd_metric(same, different, ridge):
    dimension = len(same)
    scale = float(np.trace(same) / dimension)
    if scale <= 1e-16:
        return np.zeros_like(same), dict(rank=0, scale=scale, negative_eigenvalues=0)
    regularizer = np.eye(dimension) * ridge * scale
    metric = np.linalg.inv(same + regularizer) - np.linalg.inv(different + regularizer)
    eigenvalues, eigenvectors = np.linalg.eigh((metric + metric.T) / 2)
    cutoff = max(1., float(np.max(np.abs(eigenvalues)))) * 1e-10
    positive = np.where(eigenvalues > cutoff, eigenvalues, 0.)
    transform = (eigenvectors * np.sqrt(positive)) @ eigenvectors.T
    return transform, dict(rank=int((positive > 0).sum()), scale=scale,
                           negative_eigenvalues=int((eigenvalues < -cutoff).sum()))
