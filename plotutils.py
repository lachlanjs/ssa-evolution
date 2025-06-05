import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

def plot_matrix_grid(As: list):
    # find the size of the list grid:    

    # As = [A if A is list else [A] for A in As]    

    rows = len(As)
    cols = np.max([len(As[idx]) for idx in range(len(As))])
    
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))    

    for row_idx in range(len(As)):
        for col_idx in range(len(As[row_idx])):
            A = As[row_idx][col_idx]
            ax = axs[row_idx, col_idx] if cols > 1 and rows > 1\
                else axs[col_idx] if cols > 1\
                else axs[row_idx]

            im = ax.imshow(A, cmap="inferno", norm=CenteredNorm())
            plt.colorbar(im, ax=ax)
    
    fig.show()
    return

def plot_matrix_spectra(A, B):

    A_eigs = np.linalg.eigvals(A)
    B_eigs = np.linalg.eigvals(B)

    A_eigs_real = np.real(A_eigs)
    B_eigs_real = np.real(B_eigs)
    A_eigs_imag = np.imag(A_eigs)
    B_eigs_imag = np.imag(B_eigs)

    xmin = np.min([np.min(A_eigs_real), np.min(B_eigs_real)])
    xmax = 1.0

    ymin = np.min([-5.0, np.min(A_eigs_imag), np.min(B_eigs_imag)])
    ymax = -ymin

    plt.scatter(A_eigs_real, A_eigs_imag, c="red", marker="x")
    plt.scatter(B_eigs_real, B_eigs_imag, c="blue", marker=".")

    plt.vlines([0.0], ymin=ymin, ymax=ymax, colors="black", linestyles="--")
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    plt.show()
    return