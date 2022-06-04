import numpy as np
import matplotlib.pyplot as plt
import argparse

from models import MF
from datetime import datetime


def add_command_line_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("--ratings_train_path", default="data/ratings.train.txt")
    argp.add_argument("--k_dim", default=20, type=int)
    argp.add_argument("--lr", default=0.03, type=float)
    argp.add_argument("--reg", default=0.1, type=float)
    argp.add_argument("--epochs", default=40, type=int)
    argp.add_argument("--verbose", default=1, choices=[0, 1], type=int)
    args = argp.parse_args()
    return args

def get_num_users_items(ratings_text_path):
    """
    Gets the number of users and items based on the highest index value for each.
    Assumes indexing starts at 1.
    """
    u_max_idx = -np.inf
    i_max_idx = -np.inf
    for row in open(ratings_text_path, encoding='utf8'):
        row_split = row.split()
        u_idx, i_idx = int(row_split[0]), int(row_split[1])
        if u_idx > u_max_idx:
            u_max_idx = u_idx
        if i_idx > i_max_idx:
            i_max_idx = i_idx
    return u_max_idx, i_max_idx

def plot_error_vs_epoch(losses):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses)), losses)
    ax.set_xticks(np.arange(len(losses)), labels=np.arange(len(losses)) + 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    plt.show()


if __name__ == "__main__":

    # Init command line args
    args = add_command_line_args()

    # Set random seed
    np.random.seed(13)

    # Get num users and items
    num_users, num_items = get_num_users_items(args.ratings_train_path)

    # Init MF
    mf = MF(num_users+1, num_items+1, args.k_dim)

    # Start training
    losses = []
    for epoch in range(args.epochs):

        t0 = datetime.now()

        # Train
        for row in open(args.ratings_train_path, encoding='utf8'):
            row_split = row.split()
            u_idx, i_idx, r_ui = int(row_split[0]), int(row_split[1]), int(row_split[2])
            mf.sgd_step(r_ui, u_idx, i_idx, args.lr, args.reg)

        # Compute E
        loss = 0
        for row in open(args.ratings_train_path, encoding='utf8'):
            row_split = row.split()
            u_idx, i_idx, r_ui = int(row_split[0]), int(row_split[1]), int(row_split[2])
            loss += mf.predict_loss(r_ui, u_idx, i_idx)
        loss += mf.reg_loss(args.reg)
        losses.append(loss)

        t1 = datetime.now()

        # Verbose
        if args.verbose:
            print(f"Finished epoch {epoch+1}. Error: {loss:.3f}. [{(t1-t0).seconds} S].")

    # Plot E vs iteration
    plot_error_vs_epoch(losses)
