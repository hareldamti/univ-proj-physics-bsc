import numpy as np
import scipy as sp
# A, B -> I_1 X .. X A_i X .. X B_j X .. X I_n
def to_n(n, A, i, B=None, j=-1):
    r = 1
    I = np.eye(len(A))
    for k in range(n):
        r = np.kron(r, A if k == i else B if k == j else I)
    return r

def tensor_product(As):
    r = 1
    for A in As:
        r = np.kron(r, A)
    return r

s0 = np.eye(2)
sz = np.eye(2)
sz[1][1] = -1
sx = np.ones((2,2)).astype(complex) - np.eye(2)
sy = np.zeros((2,2)).astype(complex)
sy[1][0] = 1j
sy[0][1] = -1j
splus = np.zeros((2,2)).astype(complex)
sminus = np.zeros((2,2)).astype(complex)
splus[0][1] = 1
sminus[1][0] = 1

def fill_list(x, n):
    if type(x) == float or type(x) == int or type(x) == np.float64: return np.ones(n) * x
    assert len(x) == n
    return np.array(x)

def U(H):
    evals, evecs = np.linalg.eig(H)
    m_evals, m_evecs = zip(*sorted(zip(evals, evecs), key=lambda e: -e[0]))
    m_evecs = np.array(m_evecs)
    return lambda t: evecs @ np.diag(np.exp(- 1j * t * evals)) @ evecs.T

def maj_plus (v): return np.array([v[i] + v[i + len(v) // 2] for i in range(len(v) // 2)])
def maj_minus (v): return 1j * np.array([(v[i + len(v) // 2] - v[i]) for i in range(len(v) // 2)])
def maj_ordered (v): return np.array(sum(list([list(x) for x in zip(maj_plus(v), maj_minus(v))]), []))

def power_set(n):
    if n == 0: return [[]]
    else: return sum([[j + [i] for i in [0, 1]] for j in power_set(n - 1)], [])

def c(i, n, dagger=False):
    As = [-sz if j < i else 0.5 * (sx + (1 if dagger else -1) * 1j * sy) if j == i else s0 for j in range(n)]
    return tensor_product(As)

def zero(n):
    zero = np.zeros((2 ** n, 1)).astype(np.complex128)
    zero[-1, 0] = 1
    return zero

def intersections_(U, V, ftol):
    M = np.hstack((U, -V))
    nullspace = sp.linalg.null_space(M, rcond=ftol)
    nullspace_U = nullspace[:np.shape(U)[1]]
    inter = U @ nullspace_U
    for i in range(np.shape(inter)[1]):
        inter[:,i] *= 1./np.linalg.norm(inter[:,i])
    return inter

def intersections(As, ftol=1e-3):
    inter = As[0]
    for A in As[1:]:
        inter = intersections_(inter, A, ftol)
    return inter

def canon_eigen(evals, evecs):
    n = len(evals) // 2
    idx = np.argsort(evals)
    idx = np.hstack([idx[n:], idx[n - 1::-1]])
    evals_sorted = evals[idx]
    evecs_sorted = evecs[:, idx]
    n = len(idx) // 2
    for i in range(n):
        max_val = np.argmax(np.abs(evecs_sorted[:n, i]))
        evecs_sorted[:, i + n] /= evecs_sorted[max_val, i + n] / evecs_sorted[max_val, i + n]
        max_val = np.argmax(np.abs(evecs_sorted[:n, i + n]))
        evecs_sorted[:, i + n] /= evecs_sorted[max_val + n, i] / evecs_sorted[max_val, i + n]
    return evals_sorted, evecs_sorted

def expm(A, order=2):
    orders = np.zeros((order + 1, *A.shape)).astype(complex)
    orders[0] = np.eye(A.shape[0])
    orders[1] = A
    for i in range(2, order + 1):
        orders[i] = A @ orders[i - 1] / i
    return np.sum(orders, 0)

LOSCHMIDT = "loschmidt"
LOSCHMIDT_BDG = "loschmidt_bdg"
LOSCHMIDT_TFIM = "loschmidt_tfim"
STATES = "states"

#################### 

def cmap(a):
    d = 0.1
    values = [(0, np.array([1, 0, 0, 0])), (0.5 - d, np.array([1, 0, 0, 1])), (0.5, np.array([0, 0, 0, 1])), (0.5 + d, np.array([0, 0, 1, 1])), (1, np.array([0, 0, 1, 0]))]
    i = 0
    while (a > values[i][0]): i += 1
    if i > len(values): return values[i - 1][1]
    if i == 0: return values[0][1]
    k = (a - values[i - 1][0]) * 1.0 / (values[i][0] - values[i - 1][0])
    return values[i][1] * k + values[i - 1][1] * (1 - k)