"""This class is used to visualize the excitation signals.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class Visualization():
    """Visualize the excitaion signals, including
    amplitude and phase in the frequency domian and
    signals in the time domain.
    """
    def __init__(self, freq_range: tuple, 
                 U_amp: np.ndarray,
                 U_phase: np.ndarray,
                 u: np.ndarray,
                 t_stamp: np.ndarray,
                 f_stamp: np.ndarray) -> None:
        """Initialize a instance.

        Args:
            U_amp (m x N): The amplitude in the frequency domain.
            U_phase (m x N): The random phase in the frequency domain.
            u (m x N): The signals in the time domain.
            t_stamp (array): The time stamp.
            f_stamp (array): The frequency stamp.
        """
        self.freq_range = freq_range
        self.U_amp = U_amp
        self.U_phase = U_phase
        self.u = u
        self.t_stamp = t_stamp
        self.f_stamp = f_stamp
        self.m, self.N = self.U_amp.shape
        self.idx = self.get_freq_index(freq_range, self.f_stamp)
    
    @staticmethod
    def get_freq_index(freq_range: tuple,
                       f_stamp: np.ndarray) -> tuple:
        """Get the indices of the start and end
        frequencies in the stamp.
        """        
        return (np.where(f_stamp == freq_range[0])[0][0], 
                np.where(f_stamp == freq_range[1])[0][0])
    
    @staticmethod
    def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
        """Format the axes
        """
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

    @staticmethod
    def plot_ax(ax: Axes, signal: np.ndarray, stamp: np.ndarray) -> None:
        """Plot one ax.
        """
        ax.plot(stamp, signal, linewidth=1.0, linestyle='-')

    def plot_multi_axes(self, axes: Axes, idx: int) -> None:
        """Plot multi axes.
        """
        # plot the amplitude
        ax = axes[0]
        self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Amplitude')
        self.plot_ax(ax, self.U_amp[idx, :], self.f_stamp*2*np.pi)
        ax.set_xlim(self.freq_range[0]*2*np.pi, 
                    self.freq_range[1]*2*np.pi)
        # plot the random phase
        ax = axes[1]
        self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Phase')
        self.plot_ax(ax, self.U_phase[idx, :], self.f_stamp*2*np.pi)
        ax.set_xlim(self.freq_range[0]*2*np.pi, 
                    self.freq_range[1]*2*np.pi)
        # plot the time signal
        ax = axes[2]
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
        ax.axhline(y=-1.0, color='black', linestyle='-', linewidth=1.0)
        self.set_axes_format(ax, r'Time in $s$', r'Time signal')
        self.plot_ax(ax, self.u[idx, :], self.t_stamp)
        
    def plot_signals(self, nr: int) -> None:
        """Plot the signals, 3 x min{nr, m}. The first row
        involves the amplitude in the frequency domain. The
        second row involves the phase in the frequency domian.
        The third row involves the time singals.

        Args:
            nr (int): The number of signals to plot.
        """
        nr_signals = np.min((nr, self.m))

        fig, axes = plt.subplots(3, nr_signals, figsize=(10*nr_signals, 16))
        if nr_signals == 1:
            self.plot_multi_axes(axes, 0)
        else:
            for i in range(nr_signals):
                self.plot_multi_axes(axes[:, i], i)
        
        plt.tight_layout()
        plt.show()