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

def maj_plus (v): return np.array([v[i] + v[i + len(v) // 2] for i in range(len(v) // 2)])
def maj_minus (v): return 1j * np.array([(v[i + len(v) // 2] - v[i]) for i in range(len(v) // 2)])
def maj_ordered (v): return np.array(sum(list([list(x) for x in zip(maj_plus(v), maj_minus(v))]), []))

def power_set(n):
    if n == 0: return [[]]
    else: return sum([[j + [i] for i in [0, 1]] for j in power_set(n - 1)], [])

def c(i, n, dagger=False):
    As = [-sz if j < i else 0.5 * (sx + (1 if dagger else -1) * 1j * sy) if j == i else s0 for j in range(n)]
    return tensor_product(As)

def gamma(i, n):
    j = i // 2
    if i % 2 == 0:
        return c(j, n) + c(j, n, dagger=True)
    return 1j * ( c(j, n, dagger=True) - c(j, n))

def nambu_to_spinchain(v):
    n = np.size(v) // 2
    return sum([c(i, n) * v.flatten()[i] + c(i, n, dagger=True) * v.flatten()[i + n] for i in range(n)])

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
    # for i in range(n):
    #     max_val = np.argmax(np.abs(evecs_sorted[:n, i]))
    #     evecs_sorted[:, i + n] /= evecs_sorted[max_val, i + n] / evecs_sorted[max_val, i + n]
    #     max_val = np.argmax(np.abs(evecs_sorted[:n, i + n]))
    #     evecs_sorted[:, i + n] /= evecs_sorted[max_val + n, i] / evecs_sorted[max_val, i + n]
    return evals_sorted, evecs_sorted

def expm(A, order=2):
    orders = np.zeros((order + 1, *A.shape)).astype(complex)
    orders[0] = np.eye(A.shape[0])
    orders[1] = A
    for i in range(2, order + 1):
        orders[i] = A @ orders[i - 1] / i
    return np.sum(orders, 0)

def gamma_n(n):
    Gamma = np.zeros((2 * n, 2 * n)).astype(complex)
    _block = np.ones((2, 2)).astype(complex)
    _block[0, 1] = 1j
    _block[1, 1] = -1j
    for i in range(n):
        Gamma[2 * i: 2 * (i + 1), 2 * i: 2 * (i + 1)] = _block
    Gamma = Gamma[list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))] * 0.5 ** 0.5
    return Gamma

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


def plot_complex_matrix(M,
                        ax=None,
                        vmin=None,
                        vmax=None,
                        gamma: float = 1.0,
                        show_phase_colorbar: bool = True,
                        show_amp_colorbar: bool = True,
                        cmap_phase: str = 'hsv',
                        cmap_amp: str = 'gray',
                        title: str | None = None,
                        origin: str = 'upper',
                        interpolation: str = 'nearest',
                        aspect: str | float = 'equal'):
    """
    Plot a complex matrix with brightness = abs(matrix) and hue = phase (rainbow).
    Parameters:
      M: array-like complex matrix
      ax: matplotlib Axes (optional). If None, a new figure/axes is created.
      vmin, vmax: amplitude bounds for brightness mapping. If None, automatic from data.
      gamma: gamma correction for brightness (v = (|M|-vmin)/(vmax-vmin))**gamma
      show_phase_colorbar: whether to show a vertical colorbar for phase (radians)
      show_amp_colorbar: whether to show a horizontal colorbar for amplitude |M|
      cmap_phase: colormap for phase colorbar (default 'hsv' for circular rainbow)
      cmap_amp: colormap for amplitude colorbar (default 'gray')
      title: plot title
      origin, interpolation, aspect: forwarded to imshow
    Returns:
      fig, ax, im, amp_cbar, phase_cbar
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    M = np.asarray(M)
    amp = np.abs(M)
    phase = np.angle(M)

    if vmin is None:
        vmin = float(amp.min()) if amp.size else 0.0
    if vmax is None:
        vmax = float(amp.max()) if amp.size else 1.0
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    v = np.clip((amp - vmin) / denom, 0.0, 1.0)
    if gamma != 1.0:
        v = np.clip(v ** float(gamma), 0.0, 1.0)

    # map phase [-pi, pi] -> hue [0,1)
    h = (phase + np.pi) / (2.0 * np.pi)
    s = np.ones_like(h)
    hsv = np.stack([h, s, v], axis=-1)        # shape (..., 3)
    rgb = mpl.colors.hsv_to_rgb(hsv)         # convert to RGB in 0..1

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(rgb, origin=origin, interpolation=interpolation, aspect=aspect)
    if title:
        ax.set_title(title)

    amp_cbar = None
    phase_cbar = None

    if show_phase_colorbar:
        norm_phase = Normalize(vmin=-np.pi, vmax=np.pi)
        sm_phase = mpl.cm.ScalarMappable(norm=norm_phase, cmap=cmap_phase)
        sm_phase.set_array([])  # required for colorbar
        phase_cbar = fig.colorbar(sm_phase, ax=ax, orientation='vertical', pad=0.02, fraction=0.05)
        phase_cbar.set_label('Phase (radians)')
        phase_cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        phase_cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    if show_amp_colorbar:
        norm_amp = Normalize(vmin=vmin, vmax=vmax)
        sm_amp = mpl.cm.ScalarMappable(norm=norm_amp, cmap=cmap_amp)
        sm_amp.set_array([])
        amp_cbar = fig.colorbar(sm_amp, ax=ax, orientation='horizontal', pad=0.12, fraction=0.05)
        amp_cbar.set_label('|M| (amplitude)')

    return fig, ax, im, amp_cbar, phase_cbar
