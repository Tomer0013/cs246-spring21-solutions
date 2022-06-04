import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf


def parse_row(x):
    try:
        x1, x2 = x.split('\t')
        x1 = int(x1)
        x2 = tuple([int(v) for v in x2.split(',')])
        return (x1, x2)
    except ValueError:
        x1 = int(x.split('\t')[0])
        return (x1, ())


def pair_with_each_friend(x):
    user = x[0]
    friends = x[1]
    return [(friend, user) for friend in friends]


def pairs_num_friends_to_user_and_list(x):
    u1 = (x[0][0], [(x[0][1], x[1])])
    u2 = (x[0][1], [(x[0][0], x[1])])
    return [u1, u2]


def sort_list(x):
    user = x[0]
    rec_list = sorted(x[1], key=lambda x: (-x[1], x[0]), reverse=False)
    return (user, [x[0] for x in rec_list])


def main():

    # Create the session
    conf = SparkConf().set("spark.ui.port", "4050")

    # Create the context
    sc = pyspark.SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    # Read data
    social_data = spark.read.text('data/soc-LiveJournal1Adj.txt')

    # Start map reduce process
    social_data_k_v = social_data.rdd.map(lambda x: parse_row(x.value))
    social_data_pairs = social_data_k_v.flatMap(pair_with_each_friend)
    social_data_mutual_friends_tups = social_data_pairs.join(social_data_pairs)
    social_data_mutual_friends_tups = social_data_mutual_friends_tups.filter(lambda x: x[1][0] < x[1][1])
    social_data_mutual_friends_tups = social_data_mutual_friends_tups.map(lambda x: (x[1], 1))

    # Filter out pairs that are already friends
    relevant_pairs = social_data_pairs.filter(lambda x: x[0] < x[1]).map(lambda x: (x, False))
    social_data_mutual_friends_tups = social_data_mutual_friends_tups.leftOuterJoin(relevant_pairs)
    social_data_mutual_friends_tups = social_data_mutual_friends_tups.filter(lambda x: x[1][1] != False)
    social_data_mutual_friends_tups = social_data_mutual_friends_tups.map(lambda x: (x[0], 1))

    # Continue map reduce process
    social_data_pairs_num_friends = social_data_mutual_friends_tups.reduceByKey(lambda a, b: a + b)
    social_data_user_as_key = social_data_pairs_num_friends.flatMap(pairs_num_friends_to_user_and_list)
    social_data_user_rec_list = social_data_user_as_key.reduceByKey(lambda a, b: a + b)
    social_data_user_rec_list = social_data_user_rec_list.map(sort_list)
    user_recs = social_data_user_rec_list.collect()

    # Save recs to dict
    user_to_recs_dict = {u_r_tup[0]: u_r_tup[1] for u_r_tup in user_recs}

    # Sanity check
    print(user_to_recs_dict[11][:10])


if __name__ == '__main__':
    main()
