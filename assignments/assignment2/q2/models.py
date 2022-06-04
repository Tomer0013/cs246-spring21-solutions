import numpy as np
import pyspark

class MySparkKMeans:

    def __init__(self, init_centroids, sc:pyspark.SparkContext):
        self.centroids = init_centroids
        self.sc = sc

    def fit(self, data, distance_metric, max_iter=20, verbose=False):

        assert distance_metric in ["euclidean", "manhattan"]

        if distance_metric == "euclidean":
            dist_func = lambda a,b: np.sqrt(np.sum(np.power(a-b, 2), axis=1))
            cost_func = lambda x: x**2
        if distance_metric == "manhattan":
            dist_func = lambda a,b: np.sum(np.abs(a-b), axis=1)
            cost_func = lambda x: x

        costs = []
        for i in range(max_iter):
            self.centroids, iter_cost = self._train_step(data, dist_func, cost_func)
            costs.append(iter_cost)

            if verbose:
                print(f"Finished iteration {i+1}")

        return costs

    def transform(self, data):
        raise NotImplementedError  # Implementation wasn't required by the exercise.

    def _train_step(self, data, dist_func, cost_func):

        def assign_to_cluster_with_min_dist(x, centroids, dist_func):
            dists = dist_func(x, centroids)
            assigned_cluster = np.argmin(dists)
            return x, assigned_cluster, dists[assigned_cluster]

        broadcast_centroids = self.sc.broadcast(self.centroids)
        vec_cluster_min_dist = data.map(lambda x: assign_to_cluster_with_min_dist(x, broadcast_centroids.value, dist_func))
        cluster_vec_len = vec_cluster_min_dist.map(lambda x: (x[1], (x[0], 1)))
        cluster_sum_vec_len = cluster_vec_len.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        new_centroids = np.stack(cluster_sum_vec_len.map(lambda x: x[1][0] / x[1][1]).collect())
        cost = self.sc.accumulator(0)
        vec_cluster_min_dist.foreach(lambda x: cost.add(cost_func(x[2])))
        return new_centroids, cost.value




