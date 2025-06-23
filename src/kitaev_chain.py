import numpy as np
from utils import sx, sy, sz, s0, to_n, fill_list, U, maj_ordered, cmap, LOSCHMIDT, STATES
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import rc
rc('animation', ffmpeg_path='C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe')
fps = 15
FFwriter=FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])



class kitaev_chain_model:
    def __init__(self, n: int, mu: float | list[float], t: float | list[float], delta: float | list[float]):
        self.n = n
        self.mu = fill_list(mu, n)
        self.t = fill_list(t, n)
        self.delta = fill_list(delta, n)
    
    def bdg_hamiltonian(self):
        H = np.zeros((2 * self.n, 2 * self.n))
        for i in range(self.n):
            H[i][i] = -self.mu[i]
            H[i + self.n][i + self.n] = self.mu[i]
        for i in range(self.n - 1):
            H[i][i + 1] = H[i + 1][i] = - self.t[i] / 2
            H[i + self.n][i + 1 + self.n] = H[i + 1 + self.n][i + self.n] = self.t[i] / 2
            H[i][i + 1 + self.n] = H[i + 1 + self.n][i] = - self.delta[i] / 2
            H[i + 1][i + self.n] = H[i + self.n][i + 1] = self.delta[i] / 2
        return H

    def tfim_hamiltonian(self):
        H = np.zeros((2 ** self.n, 2 ** self.n)).astype(complex)
        for i in range(self.n):
            H += self.mu[i] * to_n(self.n, sz, i)
        for i in range(self.n - 1):
            H += (self.t[i] - self.delta[i]) * to_n(self.n, sx, i, sx, i + 1)
        for i in range(self.n - 1):
            H += (self.t[i] + self.delta[i]) * to_n(self.n, sy, i, sy, i + 1)
        H *= - 1. / 2
        return H

    def tfim_hamiltonian_as_sum(self):
        H = []
        for i in range(self.n):
            H.append ( -.5 * self.mu[i] * to_n(self.n, sz, i) )
        for i in range(self.n - 1):
            H.append ( -.5 * (self.t[i] - self.delta[i]) * to_n(self.n, sx, i, sx, i + 1) )
        for i in range(self.n - 1):
            H.append ( -.5 * (self.t[i] + self.delta[i]) * to_n(self.n, sy, i, sy, i + 1) )
        return H


class quench_simulation_bdg:
    def __init__(self, model0: kitaev_chain_model, model: kitaev_chain_model):
        self.H0 = model0.bdg_hamiltonian()
        self.H = model.bdg_hamiltonian()
        self.n = model0.n
        self.U = U(self.H)
        self.simulation_data = {
            LOSCHMIDT: [],
            STATES: [],
        }
        self.solve_H0()

    def solve_H0(self):
        evals, evecs = np.linalg.eig(self.H0)
        evecs = evecs.T
        evals, evecs = zip(*sorted(zip(evals, evecs), key=lambda e: -e[0]))
        evecs = np.array(evecs)
        self.evecs0 = evecs
        self.evals0 = evals
        return self.evals0, self.evecs0
    
    def plot_initial_zero_eigenstates(self, title = "Initial eigenstates"):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        eigenpairs = list(filter(lambda pair: pair[0] ** 2 < 1e-5, zip(self.evals0, self.evecs0)))
        for pair in eigenpairs:
            ax1.plot(np.absolute(maj_ordered(pair[1])) ** 2)

    def fill_sim(self, dt, T):
        self.simulation_data = {
            LOSCHMIDT: [],
            STATES: [],
        }
        for i in range(int(T // dt)):
            Ut = self.U(dt * 1.0 * i).T
            self.simulation_data[STATES].append(self.evecs0 @ Ut.T)
            self.simulation_data[LOSCHMIDT].append(np.absolute(np.linalg.det((self.evecs0 @ Ut @ self.evecs0.T)[:self.n,:self.n])) ** 2)
    
    @property
    def frames(self): return len(self.simulation_data[LOSCHMIDT])

    def save_animation(self, title):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        x = np.linspace(-1, 1, 2 * self.n)
        plots = [None for _ in range(2 * self.n)]
        for i in range(2 * self.n):
            plots[i] = ax1.plot(x, np.absolute(maj_ordered(self.simulation_data[STATES][0][i]) ** 2),
                                c = cmap(np.real((self.evals0[i] - min(self.evals0)) / (max(self.evals0) - min(self.evals0)))))[0]
        
        frame_text = ax1.text(0.5, 0.3, "0")

        def init(): return tuple(plots)

        def animate(frame):
            for i in range(2 * self.n):
                plots[i].set_data(x, np.absolute(maj_ordered(self.simulation_data[STATES][frame][i]) ** 2))
            frame_text.set_text(f"Frame: {frame}\ncos($\\theta$) = {self.simulation_data[LOSCHMIDT][frame]:.3f}")
            return tuple(plots)
        
        anim = FuncAnimation(fig, animate, init_func=init,
                        frames = self.frames, interval = 30, blit = True)
    
        anim.save(f"simulations/{title}.mp4", writer=FFwriter)

    def plot_loschmidt(self, title = "Loschmidt plot for quench"):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        ax1.set_ylabel("Loschmidt amplitude")
        ax1.set_xlabel("Frame")
        ax1.plot(self.simulation_data[LOSCHMIDT])
        fig.show()


class quench_simulation_tfim:
    def __init__(self, model0: kitaev_chain_model, model: kitaev_chain_model):
        self.H0 = model0.tfim_hamiltonian()
        self.H = model.tfim_hamiltonian()
        self.n = model0.n
        self.U = U(self.H)
        self.simulation_data = {
            LOSCHMIDT: [],
            STATES: [],
        }
        self.solve_H0()

    def solve_H0(self):
        evals, evecs = np.linalg.eig(self.H0)
        evecs = evecs.T
        evals, evecs = zip(*sorted(zip(evals, evecs), key=lambda e: -e[0]))
        evecs = np.array(evecs)
        evals = np.real(evals)
        self.evecs0 = evecs
        self.evals0 = evals
        return self.evals0, self.evecs0

    def plot_initial_zero_eigenstates(self, title = "Initial eigenstates"):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        eigenpairs = list(filter(lambda pair: pair[0] ** 2 < 1e-5, zip(self.evals0, self.evecs0)))
        for pair in eigenpairs:
                ax1.plot([
            np.absolute(pair[1].T @ to_n(self.n, sz, i) @ pair[1]) ** 2
            for i in range(self.n)
        ])

    def fill_sim(self, dt, T):
        self.simulation_data = {
            STATES: [],
        }
        for i in range(int(T // dt)):
            Ut = self.U(dt * 1.0 * i).T
            self.simulation_data[STATES].append(self.evecs0 @ Ut.T)
    
    @property
    def frames(self): return len(self.simulation_data[STATES])

    def save_animation(self, title):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(13, 5)
        ax1.set_title(title)
        x = np.linspace(-1, 1, self.n)
        plots = [None for _ in range(2 ** self.n)]
        for i in range(2 ** self.n):
            plots[i] = ax1.plot(x, [np.absolute(self.simulation_data[STATES][0][i].T @ to_n(self.n, sz, i) @ self.simulation_data[STATES][0][i]) ** 2 for i in range(self.n)],
                                c = cmap(np.real((self.evals0[i] - min(self.evals0)) / (max(self.evals0) - min(self.evals0)))))[0]
        
        frame_text = ax1.text(0.5, 0.3, "0")

        def init(): return tuple(plots)

        def animate(frame):
            for i in range(2 ** self.n):
                plots[i].set_data(x, [np.absolute(self.simulation_data[STATES][0][i].T @ to_n(self.n, sz, i) @ self.simulation_data[STATES][0][i]) ** 2 for i in range(self.n)])
            frame_text.set_text(f"Frame: {frame}")
            return tuple(plots)
        
        anim = FuncAnimation(fig, animate, init_func=init,
                        frames = self.frames, interval = 30, blit = True)
    
        anim.save(f"simulations/{title}.mp4", writer=FFwriter)
