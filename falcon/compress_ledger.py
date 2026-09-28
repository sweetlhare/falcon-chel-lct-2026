"""Fixed 1,024-D CHEL branch composition used by production inference."""
import numpy as np

from .evidence_ledger import unit


GLOBAL_WEIGHT = 0.75
V1_SUPPORT_WEIGHT = 0.1875
V2_SUPPORT_WEIGHT = 0.0625


def exact_compact(global_embedding, support_v1, support_v2):
    return unit(np.concatenate([
        global_embedding * np.sqrt(GLOBAL_WEIGHT),
        support_v1 * np.sqrt(V1_SUPPORT_WEIGHT),
        support_v2 * np.sqrt(V2_SUPPORT_WEIGHT),
    ], axis=1))
