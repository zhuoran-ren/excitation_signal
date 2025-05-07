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

        Attributes:
            U_amp (nr_inputs x m x N): The amplitude in the frequency domain.
            U_phase (nr_inputs x m x N): The random phase in the frequency domain.
            u (nr_inputs x m x N): The signals in the time domain.
            t_stamp (array): The time stamp for each signal.
            f_stamp (array): The frequency stamp for each signal.
        """
        self.freq_range = freq_range
        self.U_amp = U_amp
        self.U_phase = U_phase
        self.u = u
        self.t_stamp = t_stamp
        self.f_stamp = f_stamp
        self.nr_inputs, self.m, self.N = self.U_amp.shape
        self.idx = self.get_freq_index(self.freq_range, 
                                       self.f_stamp)
        self.real_signal = self.get_real_signal()
        
    def get_real_signal(self):
        """From the decoupled signal to the real signal
        Inputs:
        u(nr_inputs-1 x m x N): the decoupled signal
        Retunrs:
        real_signal(nr_inputs x m x N): the real signal that excite the hardware
        """
        real_signal = np.zeros((self.nr_inputs + 1, self.m, self.N))
        p_bar = np.zeros((self.m, self.N))
        real_signal[0, :, :] = np.maximum.reduce([p_bar, p_bar + self.u[0, :, :], p_bar + self.u[0, :, :] + self.u[1, :, :]])
        real_signal[1, :, :] = np.maximum.reduce([p_bar, p_bar + self.u[1, :, :], p_bar - self.u[0, :, :]])
        real_signal[2, :, :] = np.maximum.reduce([p_bar, p_bar - self.u[1, :, :], p_bar - self.u[0, :, :] - self.u[1, :, :]])
        return real_signal

    
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

    def whether_all_positive(self):
        condition = (self.real_signal[0, :, :] > 0) & (self.real_signal[1, :, :] > 0) & (self.real_signal[2, :, :] > 0)

        if np.any(condition):
            print("There exists at least one moment when all the signals are positive")
        else:
            print("As least one signal is zero")

    def check_sum_constraint(self, u: np.ndarray, verbose: bool = True) -> bool:
        sum_u = u[0, :, :] + u[1, :, :] + u[2, :, :]
        mask = ~np.isclose(sum_u, 0)

        if np.any(mask):
            if verbose:
                print("❌ 有不满足 u[0] + u[1] + u[2] == 0 的位置：")
                print("索引：", np.where(mask))
                print("对应值：", sum_u[mask])
            return False
        else:
            if verbose:
                print("✅ 所有位置都满足 u[0] + u[1] + u[2] == 0")
            return True


    def plot_real_signal(self, axes: Axes, 
                        idx: int) -> None:
        ax = axes
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
        ax.axhline(y=-0.0, color='black', linestyle='-', linewidth=1.0)
        self.set_axes_format(ax, r'Time in $s$', r'Time signal')
        self.plot_ax(ax, self.real_signal[idx, self.idx_m, :], self.t_stamp)
        ax.axhline(y=self.real_signal[idx, self.idx_m, 0], color='red', linestyle='-', linewidth=0.5)
        ax.axhline(y=self.real_signal[idx, self.idx_m, -1], color='red', linestyle='-', linewidth=0.5)

 
    def plot_column(self, axes:Axes,
                          idx: int) -> None:
        # """plot on column of the axes"""
        # # plot the amplitude
        # ax = axes[0]
        # self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Amplitude')
        # self.plot_ax(ax, self.U_amp[idx, self.idx_m, :], self.f_stamp*2*np.pi)
        # ax.set_xlim(self.freq_range[0]*2*np.pi, 
        #             self.freq_range[1]*2*np.pi)
        # # plot the random phase
        # ax = axes[1]
        # self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Phase')
        # self.plot_ax(ax, self.U_phase[idx, self.idx_m, :], self.f_stamp*2*np.pi)
        # ax.set_xlim(self.freq_range[0]*2*np.pi, 
        #             self.freq_range[1]*2*np.pi)
        # #plot the time signal
        # ax = axes[2]
        # self.set_axes_format(ax, r'Time in $s$', r'Time signal')
        # self.plot_ax(ax, self.u[idx, self.idx_m, :], self.t_stamp)
        # ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
        # ax.axhline(y=-1.0, color='black', linestyle='-', linewidth=1.0)
        # ax.axhline(y=self.u[idx, self.idx_m, 0], color='red', linestyle='-', linewidth=0.5)
        # ax.axhline(y=self.u[idx, self.idx_m, -1], color='red', linestyle='-', linewidth=0.5)
        for i in range(3):
            ax = axes[i]
            self.set_axes_format(ax, r'Time in $s$', r'Time signal')
            self.plot_ax(ax, self.u[idx + 3*i, self.idx_m, :], self.t_stamp)
            ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
            ax.axhline(y=-1.0, color='black', linestyle='-', linewidth=1.0)
            ax.axhline(y=self.u[idx + 3*i, self.idx_m, 0], color='red', linestyle='-', linewidth=0.5)
            ax.axhline(y=self.u[idx + 3*i, self.idx_m, -1], color='red', linestyle='-', linewidth=0.5)


    def plot_signals(self, idx_m: int=1) -> None:
        """Plot one signal for all inputs. The first row
        involves the amplitude in the frequency domain. The
        second row involves the phase in the frequency domian.
        The third row involves the time singals.

        Args:
            idx_m: which signal to plot.
        """
        self.idx_m = idx_m

        fig1, axes1 = plt.subplots(3, 3, figsize=(3*self.nr_inputs, 8))

        if self.nr_inputs == 1:
            self.plot_column(axes1, 0)
        else:
            for i in range(3):
                self.plot_column(axes1[:, i], i)
        
        plt.tight_layout()
        plt.show()
        

    # def plot_multi_axes(self, axes: Axes, 
    #                     idx: int) -> None:
    #     """Plot one column of the axes.

    #     Args:
    #         axes: one column of axes
    #         idx: the idx of the column / the idx of the input channel
    #     """
    #     # plot the amplitude
    #     ax = axes[0]
    #     self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Amplitude')
    #     self.plot_ax(ax, self.U_amp[idx, self.idx_m, :], self.f_stamp*2*np.pi)
    #     ax.set_xlim(self.freq_range[0]*2*np.pi, 
    #                 self.freq_range[1]*2*np.pi)
    #     # plot the random phase
    #     ax = axes[1]
    #     self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Phase')
    #     self.plot_ax(ax, self.U_phase[idx, self.idx_m, :], self.f_stamp*2*np.pi)
    #     ax.set_xlim(self.freq_range[0]*2*np.pi, 
    #                 self.freq_range[1]*2*np.pi)
    #     # plot the time signal
    #     ax = axes[2]
    #     ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
    #     ax.axhline(y=-1.0, color='black', linestyle='-', linewidth=1.0)
    #     self.set_axes_format(ax, r'Time in $s$', r'Time signal')
    #     self.plot_ax(ax, self.u[idx, self.idx_m, :], self.t_stamp)
    #     ax.axhline(y=self.u[idx, self.idx_m, 0], color='red', linestyle='-', linewidth=0.5)
    #     ax.axhline(y=self.u[idx, self.idx_m, -1], color='red', linestyle='-', linewidth=0.5)
        
    # def plot_signals(self, idx_m: int=0) -> None:
    #     """Plot one signal for all inputs. The first row
    #     involves the amplitude in the frequency domain. The
    #     second row involves the phase in the frequency domian.
    #     The third row involves the time singals.

    #     Args:
    #         idx_m: which signal to plot.
    #     """
    #     self.idx_m = idx_m

    #     fig, axes = plt.subplots(3, self.nr_inputs, figsize=(3*self.nr_inputs, 8))
        
    #     if self.nr_inputs == 1:
    #         self.plot_multi_axes(axes, 0)
    #     else:
    #         for i in range(self.nr_inputs):
    #             self.plot_multi_axes(axes[:, i], i)
        
    #     plt.tight_layout()
    #     plt.show()