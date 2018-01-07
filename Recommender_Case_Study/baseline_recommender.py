"""
http://surprise.readthedocs.io/en/stable/building_custom_algo.html
"""
from surprise_recommender import df_from_csv, load_from_panda
import sys
import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, evaluate
from surprise_recommender import get_Iu, get_Ui

class GlobalMean(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean


class MeanofMeans(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        users = np.array([u for (u, _, _) in self.trainset.all_ratings()])
        items = np.array([i for (_, i, _) in self.trainset.all_ratings()])
        ratings = np.array([r for (_, _, r) in self.trainset.all_ratings()])

        user_means,item_means = {},{}
        for user in np.unique(users):
            user_means[user] = ratings[users==user].mean()
        for item in np.unique(items):
            item_means[item] = ratings[items==item].mean()

        self.global_mean = ratings.mean()
        self.user_means = user_means
        self.item_means = item_means

    def estimate(self, u, i):
        """
        return the mean of means estimate
        """

        if u not in self.user_means:
            return(np.mean([self.global_mean,
                            self.item_means[i]]))

        if i not in self.item_means:
            return(np.mean([self.global_mean,
                            self.user_means[u]]))

        return(np.mean([self.global_mean,
                        self.user_means[u],
                        self.item_means[i]]))

if __name__ == "__main__":
    # Load data
    ratings_df = df_from_csv('data/ratings.csv')
    data = load_from_panda(ratings_df[['userId', 'movieId', 'rating']])
    data.split(3)

    print("\nMeanOfMeans...")
    algo = MeanofMeans()
    evaluate(algo, data)

    # Query to get Predictions for User #1
    movies = [1172,2105,1953,1339,2150,1061,3671,1029,31,2455,1371,1287,1343,
                2193,1263,1293,1129,2294,2968,1405]
    for i in movies:
        print(algo.estimate(1,i))
