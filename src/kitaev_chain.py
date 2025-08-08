import numpy as np
import scipy as sp
from utils import LOSCHMIDT_BDG, LOSCHMIDT_TFIM, sx, sy, sz, s0, to_n, fill_list, maj_ordered, cmap, c, intersections, expm, LOSCHMIDT, STATES, canon_eigen
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import rc
rc('animation', ffmpeg_path='C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe')
fps = 15
FFwriter=FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])

class TV_XY_model:
    def __init__(self, L, J, gamma, h, b1, bL):
        self.L = L
        self.n = L + 2
        self.J = J
        self.gamma = gamma
        self.h = h
        self.b1 = b1
        self.bL = bL
    def H_spinchain(self, as_sum = False):
        elements = []
        for i in range(1, self.L):
            elements.append(
                self.J * .25 * (
                        to_n(self.L + 1, sx, i, sx, i + 1) * (1. + self.gamma)
                        +
                        to_n(self.L + 1, sy, i, sy, i + 1) * (1. - self.gamma)
                    )
                )
        for i in range(1, self.L + 1):
            elements.append(
                self.h * -.5 * to_n(self.L + 1, sz, i)
            )
        elements.append( 0.5 * (
            self.b1[0] * 0.5 * to_n(self.L + 1, sx, 0, sx, 1) +
            self.b1[1] * 0.5 * to_n(self.L + 1, sx, 0, sy, 1) +
            self.b1[2] * 0.5 * to_n(self.L + 1, sx, 0, sz, 1)
        ))
        elements.append( 0.5 * (
            self.bL[0] * 0.5 * to_n(self.L + 1, sx, self.L, sx, self.L + 1) +
            self.bL[1] * 0.5 * to_n(self.L + 1, sy, self.L, sx, self.L + 1) +
            self.bL[2] * 0.5 * to_n(self.L + 1, sz, self.L, sx, self.L + 1)
        ))
        return elements if as_sum else sum(elements)
    def H_bdg(self):
        H = np.zeros((2 * (self.L + 2), 2 * (self.L + 2))).astype(complex)
        for i in range(1, self.L):
            H[i, i + 1]              += -.5 * self.J
            H[i, i + 1 + self.n]     += -.5 * self.J * self.gamma
            H[i + 1, i]              += -.5 * self.J
            H[i + 1 + self.n, i]     += -.5 * self.J * self.gamma
        for i in range(1, self.L + 1):    
            H[i, i]                  += - self.h
        
        H[0, 1]          += -.5 * (self.b1[0] + 1j * self.b1[1])
        H[0, 1 + self.n] += -.5 * (self.b1[0] - 1j * self.b1[1])
        H[1, 0]          += -.5 * (self.b1[0] - 1j * self.b1[1])
        H[1 + self.n, 0] += -.5 * (self.b1[0] + 1j * self.b1[1])
        H[0][0]          +=       self.b1[2]

        H[self.L + 1, self.L]          += -.5 * (self.bL[0] + 1j * self.bL[1])
        H[self.L, self.L + 1 + self.n] += -.5 * (self.bL[0] - 1j * self.bL[1])
        H[self.L, self.L + 1]          += -.5 * (self.bL[0] - 1j * self.bL[1])
        H[self.L + 1 + self.n, self.L] += -.5 * (self.bL[0] + 1j * self.bL[1])
        H[0][0]                        += self.bL[2]

        for i in range(1, self.L):
            H[(i + 1 + self.n)%(2 * self.n), (i + self.n)%(2 * self.n)]              -= -.5 * self.J
            H[(i + 1 + self.n + self.n)%(2 * self.n), (i + self.n)%(2 * self.n)]     -= -.5 * self.J * self.gamma
            H[(i + self.n)%(2 * self.n), (i + 1 + self.n)%(2 * self.n)]              -= -.5 * self.J
            H[(i + self.n)%(2 * self.n), (i + 1 + self.n + self.n)%(2 * self.n)]     -= -.5 * self.J * self.gamma
        for i in range(1, self.L + 1):    
            H[(i + self.n)%(2 * self.n), (i + self.n)%(2 * self.n)]                  -= - self.h
        
        H[(1 + self.n)%(2 * self.n), (0 + self.n)%(2 * self.n)]          -= -.5 * (self.b1[0] + 1j * self.b1[1])
        H[(1 + self.n + self.n)%(2 * self.n), (0 + self.n)%(2 * self.n)] -= -.5 * (self.b1[0] - 1j * self.b1[1])
        H[(0 + self.n)%(2 * self.n), (1 + self.n)%(2 * self.n)]          -= -.5 * (self.b1[0] - 1j * self.b1[1])
        H[(0 + self.n)%(2 * self.n), (1 + self.n + self.n)%(2 * self.n)] -= -.5 * (self.b1[0] + 1j * self.b1[1])
        H[0][0]          -=       self.b1[2]

        H[(self.L + self.n)%(2 * self.n), (self.L + 1 + self.n)%(2 * self.n)]          -= -.5 * (self.bL[0] + 1j * self.bL[1])
        H[(self.L + 1 + self.n + self.n)%(2 * self.n), (self.L + self.n)%(2 * self.n)] -= -.5 * (self.bL[0] - 1j * self.bL[1])
        H[(self.L + 1 + self.n)%(2 * self.n), (self.L + self.n)%(2 * self.n)]          -= -.5 * (self.bL[0] - 1j * self.bL[1])
        H[(self.L + self.n)%(2 * self.n), (self.L + 1 + self.n + self.n)%(2 * self.n)] -= -.5 * (self.bL[0] + 1j * self.bL[1])
        H[0][0]                        -= self.bL[2]
        return H * .5
    


class kitaev_chain_model:
    def __init__(self, n: int, mu: float | list[float], t: float | list[float], delta: float | list[float], hasGhosts=False):
        self.hasGhosts = hasGhosts
        self.n = n
        self.N = 2 ** n
        self.mu = fill_list(mu, n)
        self.t = fill_list(t, n - 1)
        self.delta = fill_list(delta, n - 1)

        self.vac = None
        self.bdg_evals_sorted = None
        self.bdg_evecs_sorted = None

    def bdg_hamiltonian(self):
        H = np.zeros((2 * self.n, 2 * self.n))
        for i in range(self.n):
            H[i][i] = -self.mu[i] / 2
            H[i + self.n][i + self.n] = self.mu[i] / 2
        for i in range(self.n - 1):
            H[i][i + 1] = H[i + 1][i] = - self.t[i] / 2
            H[i + self.n][i + 1 + self.n] = H[i + 1 + self.n][i + self.n] = self.t[i] / 2
            H[i][i + 1 + self.n] = H[i + 1 + self.n][i] = self.delta[i] / 2
            H[i + 1][i + self.n] = H[i + self.n][i + 1] = - self.delta[i] / 2
        return H

    def bdg_eigen(self, forceEvecs = None):
        if self.hasGhosts:
            forceEvecs = np.zeros(2 * self.n)
            forceEvecs[0] = forceEvecs[self.n - 1] = forceEvecs[self.n] = 1
            forceEvecs[2 * self.n - 1] = -1
            forceEvecs *= 0.5
            forceEvecs = np.vstack([forceEvecs]).T
        H0 = self.bdg_hamiltonian()
        evals, evecs = np.linalg.eigh(H0)
        evals_sorted, evecs_sorted = canon_eigen(evals, evecs)
        if forceEvecs is not None:
            size = np.shape(forceEvecs)[1]
            evecs_sorted[:, :size] = forceEvecs
            evecs_sorted[:, self.n : self.n + size] = np.vstack([forceEvecs[self.n:], forceEvecs[:self.n]])
        self.bdg_evals_sorted = evals_sorted
        self.bdg_evecs_sorted = evecs_sorted
        P = evecs_sorted.T
        if not np.allclose(evecs_sorted @ np.diag(evals_sorted) @ evecs_sorted.T, H0):
            print("Eigenvectors numerical incompabillity")
        self.U = P[:self.n, :self.n]
        Us = P[self.n:, self.n:]
        self.V = P[:self.n, self.n:]
        Vs = P[self.n:, :self.n]

        if not (
            np.allclose(self.U.T @ self.V + self.V.T @ self.U, self.U * 0)
            and np.allclose(self.U.T @ self.U + self.V.T @ self.V, np.eye(self.n))
            and np.allclose(self.U, Us.conj()) and np.allclose(self.V, Vs.conj())
            ):
            print("U, V numerical incompabillity")

    def psi(self, i, dagger=False):
        r = sum([c(j, self.n, dagger) * self.U[i, j] + c(j, self.n, not dagger) * self.V[i, j] for j in range(self.n)], np.zeros((self.N, self.N)))
        return r.conj() if dagger else r

    def tfim_hamiltonian_JW(self):
        H = np.zeros((2 ** self.n, 2 ** self.n)).astype(complex)
        for i in range(self.n):
            H += self.mu[i] * to_n(self.n, sz, i)
        for i in range(self.n - 1):
            H += (self.t[i] - self.delta[i]) * to_n(self.n, sx, i, sx, i + 1)
        for i in range(self.n - 1):
            H += (self.t[i] + self.delta[i]) * to_n(self.n, sy, i, sy, i + 1)
        H *= -.5
        
        # H_JW = H - sigma(mu) / 2

        return H
    
    def tfim_hamiltonian_JW_on_bdg_before_split(self):
        H = np.zeros((2 ** self.n, 2 ** self.n)).astype(complex)
        for i in range(self.n):
            H += self.mu[i] * ( c(i, self.n, dagger=True) @ c(i, self.n) )
        for i in range(self.n - 1):
            H += self.t[i] * ( c(i, self.n, dagger=True) @ c(i + 1, self.n) + c(i + 1, self.n, dagger=True) @ c(i, self.n) )
            H += self.delta[i] * ( c(i, self.n) @ c(i + 1, self.n) + c(i + 1, self.n, dagger=True) @ c(i, self.n, dagger=True) )
        H *= -1
        return H

    def tfim_hamiltonian_JW_on_bdg(self):
        H = np.zeros((2 ** self.n, 2 ** self.n)).astype(complex)
        for i in range(self.n):
            H += self.mu[i] * ( c(i, self.n, dagger=True) @ c(i, self.n) - 
                               (c(i, self.n) @ c(i, self.n, dagger=True)) )
        for i in range(self.n - 1):
            H += self.t[i] * ( c(i, self.n, dagger=True) @ c(i + 1, self.n) + c(i + 1, self.n, dagger=True) @ c(i, self.n) -
                              (c(i + 1, self.n) @ c(i, self.n, dagger=True) + c(i, self.n) @ c(i + 1, self.n, dagger=True)) )
            H += self.delta[i] * ( c(i, self.n) @ c(i + 1, self.n) + c(i + 1, self.n, dagger=True) @ c(i, self.n, dagger=True) - 
                                  (c(i + 1, self.n) @ c(i, self.n) + c(i, self.n, dagger=True) @ c(i + 1, self.n, dagger=True))
                                  )
        H *= -0.5
        return H

    def tfim_vac_from_G(self, k = 1e-3):
        idx = np.where(self.bdg_evals_sorted > k)[0][0] # first positive eigenvalue
        U_prime = self.U[idx:, idx:]
        V_prime = self.V[idx:, idx:]
        G = -np.linalg.inv(U_prime) @ V_prime
        A = 0.5 * sum([sum([G[i - idx][j - idx] * c(i, self.n, True) @ c(j, self.n, True) for i in range(idx, self.n)], np.zeros((self.N, self.N))) for j in range(idx, self.n)], np.zeros((self.N, self.N)))
        zero = np.zeros((self.N, 1)).astype(np.complex128)
        zero[-1, 0] = 1
        vac = expm(A, 4) @ zero
        self.vac = vac / np.linalg.norm(vac)
        return self.vac
    
    def tfim_vac_from_intersections(self, k=1e-3):
        vac = intersections([sp.linalg.null_space(self.psi(i), rcond=k) for i in (range(1, self.n - 1) if self.hasGhosts else range(self.n))])
        self.vac = vac / np.linalg.norm(vac)
        if self.hasGhosts:
            self.vac = self.vac + self.psi(0, dagger=True) @ self.vac
        return self.vac
    
    def tfim_hamiltonian_as_sum(self):
        H = []
        for i in range(self.n):
            H.append ( -.5 * self.mu[i] * to_n(self.n, sz, i) )
        for i in range(self.n - 1):
            H.append ( -.5 * (self.t[i] - self.delta[i]) * to_n(self.n, sx, i, sx, i + 1) )
        for i in range(self.n - 1):
            H.append ( -.5 * (self.t[i] + self.delta[i]) * to_n(self.n, sy, i, sy, i + 1) )
        return H 

def U(H):
    if (type(H) == kitaev_chain_model):
        return lambda t: H.bdg_evecs_sorted @ np.diag(np.exp(- 1j * t * H.bdg_evals_sorted)) @ H.bdg_evecs_sorted.T
    
    evals, evecs = np.linalg.eig(H)
    m_evals, m_evecs = zip(*sorted(zip(evals, evecs), key=lambda e: -e[0]))
    m_evecs = np.array(m_evecs)
    return lambda t: evecs @ np.diag(np.exp(- 1j * t * evals)) @ evecs.T

class quench_simulation:
    def __init__(self, H0: kitaev_chain_model, H: kitaev_chain_model, includeTfim=True):
        self.H0 = H0
        self.H = H
        self.n = H.n
        self.U_bdg = U(self.H)
        self.includeTfim = includeTfim
        if includeTfim:
            H_tfim = H.tfim_hamiltonian_JW()
            tfim_energies, _ = np.linalg.eigh(H_tfim)
            self.U_tfim = U(0.5 * (H_tfim - tfim_energies[0] * np.eye(H.N)))

        H0.bdg_eigen()
        H.bdg_eigen()

        if includeTfim: H0.tfim_vac_from_intersections()
        self.simulation_data = {
            LOSCHMIDT_BDG: [],
            LOSCHMIDT_TFIM: [],
            STATES: [],
        }

    def plot_initial_zero_eigenstates(self, title = "Initial eigenstates"):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        eigenpairs = list(filter(lambda pair: pair[0] ** 2 < 1e-5, zip(self.H0.bdg_evals_sorted, self.H0.bdg_evecs_sorted.T)))
        for pair in eigenpairs:
            ax1.plot(np.absolute(maj_ordered(pair[1])) ** 2)

    def fill_sim(self, dt, T):
        t_range = np.arange(0, T, dt)
        L_bdg = np.array([np.abs(np.linalg.det(self.H0.bdg_evecs_sorted[:, :self.n].T.conj() @ self.U_bdg(t) @ self.H0.bdg_evecs_sorted[:, :self.n])) for t in t_range])
        L_tfim = t_range * 0
        if self.includeTfim:
            L_initial = np.hstack([self.H0.psi(i, dagger=True) @ self.H0.vac for i in range(1)])
            L_tfim = np.array([np.abs(np.linalg.det(L_initial.T.conj() @ self.U_tfim(t) @ L_initial)) for t in t_range]) ** 2
        states = np.array([self.U_bdg(t) @ self.H0.bdg_evecs_sorted for t in t_range])
        self.simulation_data = {
            LOSCHMIDT_BDG: L_bdg,
            LOSCHMIDT_TFIM: L_tfim,
            STATES: states
        }

    @property
    def frames(self): return len(self.simulation_data[LOSCHMIDT])

    def save_animation(self, title):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        x = np.linspace(-1, 1, 2 * self.n)
        plots = [None for _ in range(2 * self.n)]
        for i in range(2 * self.n):
            plots[i] = ax1.plot(x, np.max(np.absolute(maj_ordered(self.simulation_data[STATES][0][:, i])) ** 2, 4),
                                c = cmap(np.real((self.evals0[i] - min(self.evals0)) / (max(self.evals0) - min(self.evals0)))))[0]
        l_bdg = ax2.plot([0,0],[0,self.simulation_data[LOSCHMIDT_BDG][0]], label = 'BdG Loschmidt')
        l_tfim = ax2.plot([0.1,0.1],[0,self.simulation_data[LOSCHMIDT_TFIM][0]], label = 'TFIM Loschmidt')
        frame_text = ax1.text(0.5, 0.3, "0")

        def init(): return tuple(plots)

        def animate(frame):
            for i in range(2 * self.n):
                plots[i].set_data(x, np.max(np.absolute(maj_ordered(self.simulation_data[STATES][frame][:, i])) ** 2, 4))
            l_bdg.set_data([0,0],[0,self.simulation_data[LOSCHMIDT_BDG][frame]], label = 'BdG Loschmidt')
            l_tfim.set_data([0.1,0.1],[0,self.simulation_data[LOSCHMIDT_TFIM][frame]], label = 'TFIM Loschmidt')
            return tuple(plots)
        
        anim = FuncAnimation(fig, animate, init_func=init,
                        frames = self.frames, interval = 30, blit = True)
    
        anim.save(f"simulations/{title}.mp4", writer=FFwriter)