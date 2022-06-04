import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.loadtxt(path + 'features.txt', delimiter=',')
    targets = np.loadtxt(path + 'targets.txt')
    return data, targets


def plot_losses(losses_list, configs_list):
    fig, ax = plt.subplots(figsize=(20, 10))
    for losses, conf in zip(losses_list, configs_list):
        ax.plot(losses, label=f"Batch size {conf['batch_size']}")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("L(D)")
    ax.legend()
    plt.show()

