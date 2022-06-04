import numpy as np
from scipy.linalg import svd, eigh

if __name__ == "__main__":

    # Init
    m_mat = np.array([[1,2], [2,1], [3,4], [4,3]])

    # SVD
    u, sigma, v_t = svd(m_mat, full_matrices=False)

    # Compute eigenvectors
    e_vals, e_vecs = eigh(m_mat.T.dot(m_mat))
    sorted_idx = np.argsort(-e_vals)
    e_vals, e_vecs = e_vals[sorted_idx], e_vecs[:,sorted_idx]
    print(f"Evals:\n{e_vals}")
    print(f"Evecs:\n{e_vecs}")

    # v and e_vecs
    v = v_t.T
    print(f"V:\n{v}")

    # Eigenvalues and sigma
    print(f"Eigenvalues are\n{e_vals}")
    print(f"Sigma is\n{sigma}.\nSigma^2 is\n{sigma**2}")



