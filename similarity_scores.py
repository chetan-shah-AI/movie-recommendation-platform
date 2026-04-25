
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances


def calculate_similarity(v1, v2, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity([v1], [v2])[0][0]
    elif metric == "euclidean":
        return euclidean_distances([v1], [v2])[0][0]
    elif metric == "manhattan":
        return manhattan_distances([v1], [v2])[0][0]
    elif metric == "pairwise":
        return pairwise_distances([v1], [v2])[0][0]
    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'euclidean', or 'manhattan'.")


def calculate_pairwise_similarity(matrix, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity(matrix)
    elif metric == "euclidean":
        return euclidean_distances(matrix)
    elif metric == "manhattan":
        return manhattan_distances(matrix)
    elif metric == "pairwise":
        return pairwise_distances(matrix)
    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'euclidean', or 'manhattan'.")
    

if __name__ == "__main__":  
    v1 = [3.5, 2.5, 4]
    v2 = [4, 5, 1]
    v3 = [2, 1, 4]

    X = [v1, v2, v3]

    # similarity = cosine_similarity(X)
    # similarity

    # similarity_v1_v2 = calculate_similarity(v1, v2, metric="pairwise")

    # print("Pairwise similarity between v1 and v2:", similarity_v1_v2)

    similarity_score_X = calculate_pairwise_similarity(X, metric="euclidean")

    print("Pairwise euclidean similarity matrix:\n", similarity_score_X)