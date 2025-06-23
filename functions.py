import numpy as np

#a1a2..a+1a+2..
def kitaiev_chain_hamiltonian(w, delta: complex, mu: np.array):
    n = len(mu)
    H = np.zeros((2 * n, 2 * n)).astype(complex)
    for i in range(n):
        H[i][i] = mu[i] / 2
        H[i + n][i + n] = - mu[i] / 2
    for i in range(n - 1):
        H[i][i + 1] = H[i + 1][i] = -w / 2
        H[i + n][i + 1 + n] = H[i + 1 + n][i + n] = w / 2
        H[i + n][i + 1] = delta / 2
        H[i][i + 1 + n] = - delta / 2
        H[i + 1][i + n] = delta.conjugate() / 2
        H[i + n + 1][i] = - delta.conjugate() / 2
    return H

def jordan_weigner(J, g: np.array):
    return kitaiev_chain_hamiltonian(J, -J, 2 * J * g).real

#particle hole symmetry attempt
def project_phs_plus (v): return np.array([v[i] + v[i + len(v) // 2] for i in range(len(v) // 2)])
def project_phs_minus (v): return 1j * np.array([v[i] - v[i + len(v) // 2] for i in range(len(v) // 2)])
def project_maj (v): return np.array(sum(list([list(x) for x in zip(project_phs_minus(v), project_phs_plus(v))]), []))


def expm(A, order=2):
    orders = np.zeros((order + 1, *A.shape)).astype(complex)
    orders[0] = np.eye(A.shape[0])
    orders[1] = A
    for i in range(2, order + 1):
        orders[i] = A @ orders[i - 1] / i
    return np.sum(orders, 0)


#a1a+1..
"""
def kitaiev_chain_hamiltonian(w, delta: complex, mu: np.array):
    n = len(mu)
    H = np.zeros((2 * n, 2 * n)).astype(complex)
    for i in range(n):
        H[2 * i][2 * i] = -mu[i] 
    for i in range(n - 1):
        H[2 * i][2 * i + 2] = H[2 * i + 2][2 * i] = -w
        H[2 * i + 1][2 * i + 2] = delta
        H[2 * i + 2][2 * i + 1] = delta.conjugate()
    return H

def jordan_weigner(J, g: np.array):
    return kitaiev_chain_hamiltonian(J, -J, 2 * J * g).real
"""


#simulate 1 sudden change hamiltonians
def U(H):
    evals, evecs = np.linalg.eig(H)
    m_evals, m_evecs = zip(*sorted(zip(evals, evecs), key=lambda e: -e[0]))
    m_evecs = np.array(m_evecs)
    return lambda t: evecs @ np.diag(np.exp(- 1j * t * evals)) @ evecs.T



#simulate sudden changes hamiltonians
def U_(t, dt, H_steps: list[(np.array, float)]):
    raise "Unimplemented"



## CMAPs show
def show_cmaps():
    x_r = np.linspace(0, 1, 100)
    for (i, c) in list(enumerate(plt.colormaps())):
        plt.scatter(x_r,x_r * 0 + i * 0.6,c=[plt.get_cmap(c)(x) for x in x_r])
        plt.text(-0.3, i * 0.6, c)
    plt.gcf().set_size_inches(5, 45)


LOSCHMIDT = "loschmidt"
STATES = "states"