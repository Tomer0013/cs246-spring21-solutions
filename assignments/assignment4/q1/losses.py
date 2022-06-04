import numpy as np


def crossentropy_from_logits(logits, targets, return_dx=False):
    n = logits.shape[0]
    exp_logits = np.exp(logits)
    exp_logits_row_sum = exp_logits.sum(axis=1).reshape(-1, 1)
    norm_logits = exp_logits / exp_logits_row_sum
    oh_targets = np.eye(logits.shape[1])[targets.astype(int)]
    cross_sum = (-np.log(norm_logits) * oh_targets).sum()
    loss = cross_sum / n
    if not return_dx:
        return loss
    d_loss = -np.ones_like(logits) / (norm_logits[oh_targets.astype(bool)].reshape(-1, 1) * n)
    d_softmax = -exp_logits * exp_logits[oh_targets.astype(bool)].reshape(-1, 1)
    d_softmax /= np.power(exp_logits_row_sum, 2)
    d_softmax = np.where(oh_targets.astype(bool), d_softmax +
                         norm_logits[oh_targets.astype(bool)].reshape(-1, 1), d_softmax)
    d_loss *= d_softmax
    return loss, d_loss


if __name__ == "__main__":

    x = np.random.rand(100, 2)
    y = np.random.choice([0, 1], 100)
    f, dx = crossentropy_from_logits(x, y, True)

    # Assert loss
    assert np.isclose(f, np.log(x.shape[1]), rtol=1e-5, atol=1e-1)
    print("Loss eval test passed.")

    # Compute dx of loss by definition
    dx_by_definition = np.zeros_like(dx)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            h = 1e-8
            add = np.zeros_like(x)
            add[i, j] = h
            f_h = crossentropy_from_logits(x+add, y)
            dx_by_definition[i, j] = (f_h - f) / h

    # Assert d_loss
    assert np.allclose(dx, dx_by_definition)
    print("Gradient of loss eval test passed.")
