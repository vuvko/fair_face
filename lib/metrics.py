import numpy as np


def norm(emb):
    return np.sqrt(np.sum(emb ** 2))


def cosine(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


def euclidean(emb1, emb2):
    return -norm(emb1 - emb2)
