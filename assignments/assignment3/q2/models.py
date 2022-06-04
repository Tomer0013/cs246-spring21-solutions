import numpy as np
import pyspark
from scipy.sparse import csr_matrix


class PageRank:

    def __init__(self, data:pyspark.RDD, sc:pyspark.SparkContext, sep="\t"):

        def col_score_list_to_array(col_score_list, node_to_idx_dict):
            array = np.zeros(len(node_to_idx_dict))
            for col_score in col_score_list:
                array[node_to_idx_dict[col_score[0]]] = col_score[1]
            return csr_matrix(array)

        # Init spark sontext
        self.sc = sc

        # Init the ranks r vector
        unique_nodes_prep = data.map(lambda x: int(x.split(sep)[0]))
        unique_nodes = sorted(unique_nodes_prep.distinct().collect())
        node_to_idx_dict = {node: idx for idx, node in enumerate(unique_nodes)}
        self.idx_to_node_dict = {idx: node for idx, node in enumerate(unique_nodes)}
        self.r_vec = np.ones(len(unique_nodes)) / len(unique_nodes)
        self.n_nodes = len(node_to_idx_dict)

        # Compute the M probability matrix
        node_to_idx_dict_broadcast = sc.broadcast(node_to_idx_dict)
        node_to_node = data.map(lambda x: tuple([int(node) for node in x.split(sep)])).distinct()
        node_out = node_to_node.map(lambda x: (x[0], 1))
        node_out_count = node_out.reduceByKey(lambda a, b: a + b)
        node_node_outcount = node_to_node.join(node_out_count)
        node_node_score = node_node_outcount.map(lambda x: (x[0], (x[1][0], 1 / x[1][1])))
        row_col_score = node_node_score.map(lambda x: (x[1][0], [(x[0], x[1][1])]))
        row_col_score_list = row_col_score.reduceByKey(lambda a, b: a + b)
        #row_col_score_set = row_col_score_list.map(lambda x: (x[0], list(set(x[1]))))
        self.m_mat = row_col_score_list.map(lambda x: (x[0], col_score_list_to_array(x[1], node_to_idx_dict_broadcast.value)))

    def fit(self, iterations, teleport_beta, verbose=False):
        teleport_beta_broadcast = self.sc.broadcast(teleport_beta)
        for i in range(iterations):
            self.r_vec = self._iteration_step(teleport_beta_broadcast)
            if verbose:
                print(f"Finished iteration {i+1}")

    def _iteration_step(self, teleport_beta_broadcast):

        def csr_vec_dot_r_vec(csr_vec, r_vec, beta, n):
            dot_product = csr_vec.dot(r_vec)[0]
            return beta * dot_product + (1 - beta) * (1 / n)

        r_broadcast = self.sc.broadcast(self.r_vec)
        n_nodes_broadcast = self.sc.broadcast(self.n_nodes)
        row_dot_r = self.m_mat.map(lambda x: (x[0], csr_vec_dot_r_vec(x[1],
                                              r_broadcast.value, teleport_beta_broadcast.value, n_nodes_broadcast.value)))
        return np.array([pair[1] for pair in sorted(row_dot_r.collect(), key=lambda x: x[0])])

    def get_nodes_and_scores(self):
        return [(self.idx_to_node_dict[idx], score) for idx, score in enumerate(self.r_vec)]


class HITS:

    def __init__(self, data:pyspark.RDD, sc:pyspark.SparkContext, sep="\t"):

        def col_score_list_to_array(col_score_list, node_to_idx_dict):
            array = np.zeros(len(node_to_idx_dict))
            for col_score in col_score_list:
                array[node_to_idx_dict[col_score[0]]] = col_score[1]
            return csr_matrix(array)

        # Init spark sontext
        self.sc = sc

        # Init the h vector
        unique_nodes_prep = data.map(lambda x: int(x.split(sep)[0]))
        unique_nodes = sorted(unique_nodes_prep.distinct().collect())
        node_to_idx_dict = {node: idx for idx, node in enumerate(unique_nodes)}
        self.idx_to_node_dict = {idx: node for idx, node in enumerate(unique_nodes)}
        self.n_nodes = len(node_to_idx_dict)
        self.h_vec = np.ones(len(unique_nodes))
        self.a_vec = None

        # Compute the L matrix
        node_to_idx_dict_broadcast = sc.broadcast(node_to_idx_dict)
        node_to_node = data.map(lambda x: tuple([int(node) for node in x.split(sep)])).distinct()
        row_col_score_list = node_to_node.map(lambda x: (x[0], [(x[1], 1)])).reduceByKey(lambda a, b: a + b)
        col_row_score_list = node_to_node.map(lambda x: (x[1], [(x[0], 1)])).reduceByKey(lambda a, b: a + b)
        self.l_mat = row_col_score_list.map(
            lambda x: (x[0], col_score_list_to_array(x[1], node_to_idx_dict_broadcast.value)))
        self.l_mat_t = col_row_score_list.map(
            lambda x: (x[0], col_score_list_to_array(x[1], node_to_idx_dict_broadcast.value)))

    def fit(self, iterations, verbose=False):

        for i in range(iterations):
            self.h_vec, self.a_vec = self._iteration_step()
            if verbose:
                print(f"Finished iteration {i+1}")

    def _iteration_step(self):

        h_broadcast = self.sc.broadcast(self.h_vec)
        l_t_h = self.l_mat_t.map(lambda x: (x[0], x[1].dot(h_broadcast.value)[0]))
        a_vec = np.array([pair[1] for pair in sorted(l_t_h.collect(), key=lambda x: x[0])])
        a_vec = a_vec / np.max(a_vec)
        a_broadcast = self.sc.broadcast(a_vec)
        l_a = self.l_mat.map(lambda x: (x[0], x[1].dot(a_broadcast.value)[0]))
        h_vec = np.array([pair[1] for pair in sorted(l_a.collect(), key=lambda x: x[0])])
        h_vec = h_vec / np.max(h_vec)
        return h_vec, a_vec

    def get_nodes_and_scores(self):
        return [(self.idx_to_node_dict[idx], score) for idx, score in enumerate(self.h_vec)], \
               [(self.idx_to_node_dict[idx], score) for idx, score in enumerate(self.a_vec)]