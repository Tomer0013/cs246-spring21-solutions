import numpy as np


class MF:

    def __init__(self, num_users, num_items, k_dim):
        self.u_emb = np.random.rand(num_users, k_dim) * np.sqrt(5 / k_dim)
        self.i_emb = np.random.rand(num_items, k_dim) * np.sqrt(5 / k_dim)

    def __call__(self, u_idx, i_idx):
        return self.u_emb[u_idx].dot(self.i_emb[i_idx])

    def predict_loss(self, r_ui, u_idx, i_idx):
        return (r_ui - self(u_idx, i_idx))**2

    def reg_loss(self, reg):
        return reg * (np.sum(np.power(self.u_emb, 2)) + np.sum(np.power(self.i_emb, 2)))

    def sgd_step(self, r_ui, u_idx, i_idx, lr, reg):
        old_pu = self.u_emb[u_idx]
        old_qi = self.i_emb[i_idx]
        self.u_emb[u_idx] = old_pu - lr * (-2 * old_qi * (r_ui - self(u_idx, i_idx)) + 2 * reg * old_pu)
        self.i_emb[i_idx] = old_qi - lr * (-2 * old_pu * (r_ui - self(u_idx, i_idx)) + 2 * reg * old_qi)