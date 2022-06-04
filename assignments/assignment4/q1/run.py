import numpy as np

from utils import load_data, plot_losses
from models import Perceptron
from losses import crossentropy_from_logits
from datetime import datetime

if __name__ == "__main__":

    # Set random seed
    np.random.seed(1337)

    # Load data
    path = 'data/'
    data, targets = load_data(path)

    # Init vars
    input_dim = data.shape[1]
    n_classes = len(set(targets))
    loss_stop_threshold = 0.4

    # Init train params
    param_config_list = [
        {'batch_size': 1, 'lr': 0.1},
        {'batch_size': 20, 'lr': 0.1},
        {'batch_size': data.shape[0], 'lr': 0.25}
    ]

    # Train
    losses_list = []
    for idx, conf in enumerate(param_config_list):
        iteration = 0
        loss_vals = []
        perceptron = Perceptron(input_dim, n_classes)
        current_loss = crossentropy_from_logits(perceptron(data), targets)
        t0 = datetime.now()
        while current_loss >= loss_stop_threshold:

            # Iter count for verbose
            iteration += 1

            # Select random samples from train set
            batch_idxs = np.random.choice(np.arange(data.shape[0]), size=conf['batch_size'], replace=False)
            batch_x = data[batch_idxs]
            batch_y = targets[batch_idxs]

            # Forward pass
            batch_logits = perceptron(batch_x)

            # Backwards pass
            _, batch_d_loss = crossentropy_from_logits(batch_logits, batch_y, True)

            # Update weights
            perceptron.W -= conf['lr'] * batch_x.T.dot(batch_d_loss)
            perceptron.b -= conf['lr'] * batch_d_loss.sum(axis=0)

            # Current loss
            current_loss = crossentropy_from_logits(perceptron(data), targets)
            loss_vals.append(current_loss)

        t1 = datetime.now()
        print(f"Experiment {idx+1} finished. Iterations: {iteration}. Time elapsed: {(t1-t0).total_seconds():.3f} seconds.")
        losses_list.append(loss_vals)

    # Plot losses
    plot_losses(losses_list, param_config_list)
