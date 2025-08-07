import ot
import numpy as np

def wasserstein_distance(embeddings_main, embeddings_to_compare):
    M = ot.dist(embeddings_main, embeddings_to_compare, metric='euclidean')
    a = np.ones(len(embeddings_main)) / len(embeddings_main)
    b = np.ones(len(embeddings_to_compare)) / len(embeddings_to_compare)
    W_dist = ot.emd2(a, b, M, numItermax=100000)
    return W_dist