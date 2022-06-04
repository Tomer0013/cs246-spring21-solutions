import numpy as np
import matplotlib.pyplot as plt
import pyspark

from pyspark.sql import SparkSession
from pyspark import SparkConf
from models import MySparkKMeans


def preprocess_raw_data(text_data):
    rdd = text_data.rdd.map(lambda row: np.array([float(num) for num in row.value.split()]))
    return rdd

def plot_costs(list_of_costs, list_of_labels, y_label):
    fig, ax = plt.subplots()
    for costs, label in zip(list_of_costs, list_of_labels):
        ax.plot(np.arange(len(costs)), costs, label=label)
    ax.set_xticks(np.arange(len(costs)), labels=np.arange(len(costs))+1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()

if __name__ == "__main__":

    # Create the session
    conf = SparkConf().set("spark.ui.port", "4050")

    # Create the context
    sc = pyspark.SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    # Read data
    data_raw = spark.read.text("data/data.txt")
    c1 = np.loadtxt("data/c1.txt")
    c2 = np.loadtxt("data/c2.txt")

    # Prep raw data
    data = preprocess_raw_data(data_raw)

    # Fit kmeans
    kmeans_c1 = MySparkKMeans(c1, sc)
    costs_c1 = kmeans_c1.fit(data, "euclidean", verbose=True)
    kmeans_c2 = MySparkKMeans(c2, sc)
    costs_c2 = kmeans_c2.fit(data, "euclidean", verbose=True)

    # a-1 plots
    list_of_costs = [costs_c1, costs_c2]
    list_of_labels = ["c1 init", "c2 init"]
    plot_costs(list_of_costs, list_of_labels, "Cost (Euclidean)")
    
    # a-2
    print("Euclidean:")
    print(f"Percentage change in cost after 10 iterations with c1 init: {(costs_c1[0]-costs_c1[10])/costs_c1[0]:.5f}")
    print(f"Percentage change in cost after 10 iterations with c2 init: {(costs_c2[0] - costs_c2[10]) / costs_c2[0]:.5f}\n")

    # Fit kmeans with manhattan dist and cost
    kmeans_c1 = MySparkKMeans(c1, sc)
    costs_c1 = kmeans_c1.fit(data, "manhattan", verbose=True)
    kmeans_c2 = MySparkKMeans(c2, sc)
    costs_c2 = kmeans_c2.fit(data, "manhattan", verbose=True)

    # b-1 plots
    list_of_costs = [costs_c1, costs_c2]
    list_of_labels = ["c1 init", "c2 init"]
    plot_costs(list_of_costs, list_of_labels, "Cost (Manhattan)")

    # b-2
    print("Manhattan:")
    print(f"Percentage change in cost after 10 iterations with c1 init: {(costs_c1[0]-costs_c1[10])/costs_c1[0]:.5f}")
    print(f"Percentage change in cost after 10 iterations with c2 init: {(costs_c2[0] - costs_c2[10]) / costs_c2[0]:.5f}")
