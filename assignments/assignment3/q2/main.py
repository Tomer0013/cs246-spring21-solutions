import pyspark

from pyspark.sql import SparkSession
from pyspark import SparkConf
from models import PageRank, HITS


def print_top_k_nodes(nodes_scores_list, title, bottom=False, k=5):
    print(f"\nTop {k} nodes - " + title)
    if bottom:
        for idx, node_score in enumerate(nodes_scores_list[:5]):
            print(f"{idx+1}. Node {node_score[0]} ({node_score[1]:.6f})")
    else:
        for idx, node_score in enumerate(reversed(nodes_scores_list[-5:])):
            print(f"{idx+1}. Node {node_score[0]} ({node_score[1]:.6f})")


if __name__ == "__main__":

    # Create the session
    conf = SparkConf().set("spark.ui.port", "4050")

    # Create the context
    sc = pyspark.SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    # Read data
    data_full = sc.textFile("data/graph-full.txt")

    # Fit PageRank
    page_rank = PageRank(data_full, sc)
    page_rank.fit(40, 0.8, verbose=False)

    # Print top 5 and bottom 5 nodes
    nodes_scores_list = sorted(page_rank.get_nodes_and_scores(), key=lambda x:x[1])
    print_top_k_nodes(nodes_scores_list, "Lowest PageRank scores:", bottom=True)
    print_top_k_nodes(nodes_scores_list, "Highest PageRank scores:")

    # Fit HITS
    hits = HITS(data_full, sc)
    hits.fit(40, verbose=False)

    # Get hubbiness and authority score vectors
    h_vec_scores_list, a_vec_scores_list = hits.get_nodes_and_scores()
    h_vec_scores_list = sorted(h_vec_scores_list, key=lambda x: x[1])
    a_vec_scores_list = sorted(a_vec_scores_list, key=lambda x: x[1])

    # Print top 5 and bottom 5 noddes of hubbiness
    print_top_k_nodes(h_vec_scores_list, "Lowest hubbiness:", bottom=True)
    print_top_k_nodes(h_vec_scores_list, "Highest hubbiness:")
    print_top_k_nodes(a_vec_scores_list, "Lowest authority:", bottom=True)
    print_top_k_nodes(a_vec_scores_list, "Highest authority:")
