import pandas as pd

# from data_pipeline import create_user_movie_matrix
import config

import model

# from similarity_scores import calculate_pairwise_similarity




# user_movie_matrix = create_user_movie_matrix()

# print("User-Movie Matrix:", user_movie_matrix)

# similarity_matrix = calculate_pairwise_similarity(user_movie_matrix.values, metric="cosine")
# print("Similarity Matrix:", similarity_matrix)

# similarity_matrix_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
# print("Similarity Matrix DataFrame:\n", similarity_matrix_df)



ratings = pd.read_csv(config.RATINGS_DATASET_PATH)[:1000000]
model_weights, testset = model.train_svd_model(ratings)

model.evaluate_model(model_weights, testset )


model.save_model(model_weights)