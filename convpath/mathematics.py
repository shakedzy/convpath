import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim
from .package_types import Embedding


def cosine_similarity(vector1: Embedding, vector2: Embedding) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vector1 (Embedding): The first vector to compute the cosine similarity between.
        vector2 (Embedding): The second vector to compute the cosine similarity between.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return sk_cos_sim(np.array([vector1]), np.array([vector2]))[0][0]


def dynamic_time_warping_distance(path1: list[Embedding], path2: list[Embedding]) -> float:
    """
    Computes the dynamic time warping distance between two paths.

    Args:
        path1 (list[Embedding]): The first path to compute the dynamic time warping distance between.
        path2 (list[Embedding]): The second path to compute the dynamic time warping distance between.

    Returns:
        float: The dynamic time warping distance between the two paths.
    """
    n, m = len(path1), len(path2)
    cost_matrix = cdist(path1, path2)
    
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            dtw_matrix[i, j] = cost_matrix[i-1, j-1] + min(dtw_matrix[i-1, j], 
                                                           dtw_matrix[i, j-1], 
                                                           dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]
