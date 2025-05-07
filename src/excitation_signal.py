"""Classes for generating the excitation signals
with random phases in the frequency domain.
"""
import numpy as np
from tqdm import tqdm

np.random.seed(42)

class ExcitationSignal():
    """Generate m different excitation signals with 
    random phases in the frequency domain, and each 
    signal repeats p times.
    """
    def __init__(self, freq_range: tuple=(0.0, 5.0),
                 f: float=100.0, 
                 N: int=100, 
                 amp: int=100.0,
                 eps=1e-1) -> None:
        """Initializa a instance.

        Args:
            fre_range (tuple): The excited frequency range.
            f (flaot): The sampling frequency.
            N (int): The number of points for each signal.
            amp (float): Amplitude of the signal.
            eps (float): The tolerance.
        """
        self.freq_range = freq_range
        self.f = f
        self.N = N
        
        self.amp = amp
        self.eps = eps

        self.M = np.array([
                             [np.cos(np.deg2rad(0)),     np.sin(np.deg2rad(0))],
                             [np.cos(np.deg2rad(120)),   np.sin(np.deg2rad(120))],
                             [np.cos(np.deg2rad(-120)),  np.sin(np.deg2rad(-120))]])

        self.initialization(self.freq_range, 
                            self.f, 
                            self.N)

        
    def initialization(self, freq_range: tuple,
                       f: float, 
                       N: int) -> None:
        """Initialize the necessary parameters.

        Args:
            freq_range (tuple): The excited frequency range.
            f (float): The sampling frequency.
            N (int): The number of points of each signal.
        
        Attributes:
            df (float): The sampling interval in the frequency domain.
            T (float): The execution time for each signal.
            t_stamp (array): The time stamp.
            f_stamp (array): The frequency stamp.
            idx[0] (int): The index of the start of the frequency range in the frequency stamp.
            idx[1] (int): The index of the end of the frequency range in the frequency stamp.
        """
        self.df = self.get_sampling_interval(f, N)
        self.T = self.get_total_time(self.df)
        self.t_stamp = self.get_time_stamp(f, N)
        self.f_stamp = self.get_freq_stamp(f, N)
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
    def get_freq_stamp(f: float, 
                       N: int) -> np.ndarray:
        """Calculate the frequency stamp: f_i = i*(f/N).
        """
        return np.arange(0, N) / N * f
        
    @staticmethod
    def get_time_stamp(f: float, 
                       N: float) -> np.ndarray:
        """Calculate the time stamp: t_i = i/f.
        """
        return np.arange(0, N) / f

    @staticmethod
    def get_total_time(df: float) -> float:
        """Calculate the execution time for a signal:
        T = 1 / df.
        
        Args:
            df (float): The sampling interval in the frequency domain.
        
        Returns:
            T (float): The execution time for a signal.
        """
        return 1 / df

    @staticmethod
    def get_sampling_interval(f: float, 
                              N: int) -> float:
        """Calculate the sampling interval in the 
        frequency domain: df = f / N.

        Args:
            f (float): The sampling frequency.
            N (int): The number of points of a signal.

        Returns:
            df (float): The sampling interval in the frequency domain.
        """
        return f / N

    @staticmethod
    def _get_frequency_amplitude(N: int, 
                                 amp: float, 
                                 idx: tuple) -> np.ndarray:
        """Get the amplitude in the frequency domain.
        """
        # initialize frequency domain amplitude (zero array)
        U_amp = np.zeros(N)
        # assign amplitude to selected frequency range
        idx_vector = np.arange(idx[0], idx[1] + 1)
        U_amp[idx_vector] = amp
        # set DC component to 0
        U_amp[0] = 0
        return U_amp
    
    @staticmethod
    def _get_random_phase(N: int) -> np.ndarray:
        """
        Generate a random phase array: phi \in [0, 2\pi].

        Args:
            N (int): Number of points of a signal.

        Returns:
            phi (array): Random phase values in radians.
        """
        return np.random.rand(N) * 2 * np.pi
    
    @staticmethod
    def _get_complex_signal(U_amp: np.ndarray,
                            U_phase: np.ndarray) -> complex:
        """Generate the complex signal.
        """
        return U_amp * np.exp(1j * U_phase)

    def get_frequency_signal(self, N: int, 
                             amp: float, 
                             idx: tuple) -> tuple[complex,
                                                  np.ndarray,
                                                  np.ndarray]:
        """Get one frequency signal with random phase.

        Args:
            N: the number of points
            amp: the amplitude in the frequency domain
            idx: the start and end indices of the excited range. 
        
        Returns:
            U (complex): The frequency signal.    
            U_amp (array): The amplitude in the frequency domain.
            U_phase (array): The phase in the frequency domain.
        """
        U_amp = self._get_frequency_amplitude(N, amp, idx)
        U_phase = self._get_random_phase(N)
        U = self._get_complex_signal(U_amp, U_phase)
        return U, U_amp, U_phase

    @staticmethod
    def get_time_signal(U: complex):
        """Convert frequency signal to time signal
        using inverse fast Fourier transformation.

        Args:
            U (complex): The frequency signal.
        
        Returns:
            u (array): The corresponding time signal.
        """
        return np.real(np.fft.ifft(U))

    @staticmethod
    def _get_normalization(u: np.ndarray) -> np.ndarray:
        """Normalize the signal wrt the largest abs. value.
        """
        return u/np.max(np.abs(u))

    def get_signals_orthogonal(self, N: int, 
                    amp: int, 
                    nr_inputs: int,
                    idx: tuple) -> tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
        """For each input channel, generate one signal
        according to the mode (orthogonal or random). Ensuring 
        that the difference between the beginning and the end 
        of the same signal is not too large (< eps).
        
        Args:
            N: the number of points.
            amp: the amplitude
            nr_inputs: the number of inputs
            idx: the start and end indices
            mode: the way to generate signals for different inputs

        Returns:
            U_amp (nr_inputs x N): The amplitude in the frequency domain.
            U_phase (nr_inputs x N): The random phase in the frequency domain.
            u (nr_inputs x N): The time signal.
        """
        U_amp = np.zeros((nr_inputs, N))
        U_phase = np.zeros((nr_inputs, N))
        u = np.zeros((nr_inputs, N))
        u_original = np.zeros((6, N))

        for i in range(3):
            # For each segment of the arm
            _U_amp, _U_phase = self.get_one_feasible_signal(N, amp, idx)
            group_indices = [3 * i, 3 * i + 1, 3 * i + 2]
            for j in range(3):
                # For each string of one segment
                phase_shift = 2 * np.pi * j / 3
                U_amp[3*i + j, :] = _U_amp
                U_phase[3*i + j, :] = _U_phase + phase_shift
                U = self._get_complex_signal(U_amp[3*i + j, :], U_phase[3*i + j, :])
                u[3*i + j, :] = self.get_time_signal(U)

            max_val = np.max(np.abs(u[group_indices, :]))
            u[group_indices, :] /= max_val
            
        selected_indices = [0, 1, 3, 4, 6, 7]
        u_original = u[selected_indices, :]

        return U_amp, U_phase, u, u_original
    
    def get_signals_mapping(self, N: int, 
                    amp: int, 
                    idx: tuple) -> tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
        
        #Initialization
        U_amp = np.zeros((6, N))
        U_phase = np.zeros((6, N))
        u = np.zeros((9, N))
        u_original = np.zeros((6, N))

        # for i in range(3):
        #     U_amp[2*i, :], U_phase[2*i, :] = self.get_one_feasible_signal(N, amp, idx)
        #     Bx_f = self._get_complex_signal(U_amp[2*i, :], U_phase[2*i, :])
        #     u_original[2*i, :] = self._get_normalization(self.get_time_signal(Bx_f))
        #     U_amp[2*i+1, :], U_phase[2*i+1, :] = self.get_one_feasible_signal(N, amp, idx)
        #     By_f = self._get_complex_signal(U_amp[2*i+1, :], U_phase[2*i+1, :])
        #     u_original[2*i+1, :] = self._get_normalization(self.get_time_signal(By_f))
        #     u[3*i:3*i+3, :] = self.M @ np.vstack([u_original[2*i, :], u_original[2*i+1, :]])
        # return U_amp, U_phase, u, u_original
        for i in range(3):
            U_amp[2*i, :], U_phase[2*i, :] = self.get_one_feasible_signal(N, amp, idx)
            Bx_f = self._get_complex_signal(U_amp[2*i, :], U_phase[2*i, :])
            u_original[2*i, :] = self._get_normalization(self.get_time_signal(Bx_f)) * 1000
            U_amp[2*i+1, :] = U_amp[2*i, :]
            U_phase[2*i+1, :] = U_phase[2*i, :] + np.pi / 2
            By_f = self._get_complex_signal(U_amp[2*i+1, :], U_phase[2*i+1, :])
            u_original[2*i+1, :] = self._get_normalization(self.get_time_signal(By_f)) * 1000
            u[3*i:3*i+3, :] = self.M @ np.vstack([u_original[2*i, :], u_original[2*i+1, :]])
        return U_amp, U_phase, u, u_original
    # def get_signals_mapping(self, N: int, 
    #                 amp: int, 
    #                 idx: tuple) -> tuple[np.ndarray,
    #                                     np.ndarray,
    #                                     np.ndarray,
    #                                     np.ndarray]:
    #     base_amp, base_phase = self.get_one_feasible_signal(N, amp, idx)

    #     U_amp = np.zeros((6, N))
    #     U_phase = np.zeros((6, N))
    #     u_original = np.zeros((6, N))

    #     for i in range(6):
    #         phase_shift = 2 * np.pi * i / 6
    #         U_amp[i, :] = base_amp
    #         U_phase[i, :] = base_phase + phase_shift

    #         U_complex = self._get_complex_signal(U_amp[i, :], U_phase[i, :])
    #         # u_original[i, :] = self._get_normalization(self.get_time_signal(U_complex))
    #         u_original[i, :] = self.get_time_signal(U_complex)
        
    #     u = np.zeros((9, N))
    #     for i in range(3):
    #         Bx = u_original[2*i, :]
    #         By = u_original[2*i+1, :]
    #         mapped = self.M @ np.vstack([Bx, By])
    #         u[3*i:3*i+3, :] = mapped
    #     return U_amp, U_phase, u, u_original

    def get_one_feasible_signal(self,
                                N: int,
                                amp: float,
                                idx: tuple) -> tuple[np.ndarray,
                                                     np.ndarray]:
        """Generate one feasible signal, return the amplitude
        and phase. Here feasible means that it's a cyclic signal.

        Args:
            N: the number of points
            amp: the amplitude in the frequency domain
            idx: the start and end indices
        
        Returns:
            U_amp: The amplitude in the frequency domain.
            U_phase: The random phase in the frequency domain.
        """
        diff = 1000.0
        while diff > self.eps:
            U, U_amp, U_phase = self.get_frequency_signal(N, amp, idx)
            u = self._get_normalization(self.get_time_signal(U))
            diff = np.abs(u[0] - u[-1])
        return U_amp, U_phase

    def get_multi_signals(self, m: int, 
                          p: int,
                          nr_inputs: int,
                          mode: str) -> tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
        """For each input channel, generate m different signals, 
        and each signal repeats p times. Each signal is mutually
        orthogonal or totally random.

        Args:
            m (int): The number of different signals.
            p (int): Each signal repeats p times.
            nr_inputs (int): The number of inputs of the system.
            mode (str): The way to generate signals for different inputs, orthogonal or random
        
        Returns:
            U_amp (nr_inputs x m x N): The amplitude in the frequency domain of different signals.
            U_Phase (nr_inputs x m x N): The random phase in the frequency domain of differnt signals.
            u (nr_inputs x m x N): The different signals in the time domain.
            us (nr_inputs x m*N*p): The m different signals repeated p times.
            u_original(6 x m xp): The original siginal that is truly used in system identification.
        """

        u = np.zeros((nr_inputs, m, self.N))
        us = np.zeros((nr_inputs, m*p*self.N))
        u_original = np.zeros((6, m, self.N))
        if mode == 'orthogonal':
            U_amp = np.zeros((nr_inputs, m, self.N))
            U_phase = np.zeros((nr_inputs, m, self.N))
            for i in tqdm(range(m)):
                U_amp[:, i, :], U_phase[:, i, :], u[:, i, :], u_original[:, i, :] = self.get_signals_orthogonal(self.N, 
                                                                            self.amp, 
                                                                            nr_inputs,
                                                                            self.idx)
        if mode == 'mapping':
            U_amp = np.zeros((6, m, self.N))
            U_phase = np.zeros((6, m, self.N))
            for i in tqdm(range(m)):
                U_amp[:, i, :], U_phase[:, i, :], u[:, i, :], u_original[:, i, :] = self.get_signals_mapping(self.N, 
                                                                            self.amp, 
                                                                            self.idx)
        us = self.get_repeat_signals(u, p)
        u_sysid = self.get_repeat_signals(u_original, p)
        for i in range(6):
            for j in range(i + 1, 6):
                dot = np.dot(u_sysid[i, :], u_sysid[j, :])
                print(f"dot(u[{i}], u[{j}]) = {dot:.2f}")

        return U_amp, U_phase, u, us, u_sysid
    
    @staticmethod
    def get_repeat_signals(u: np.ndarray,
                           p: int) -> np.ndarray:
        """Repeat the signal p times.

        Args:
            u (nr_inputs x m x N): the time signal
            p: the number of repeat times
        
        Returns:
            us (nr_inputs x m*p*N): repeated time signal
        """
        nr_inputs, m, N = u.shape
        us = np.tile(u, (1, 1, p))  # (nr_inputs, m*p, N)
        return us.reshape(nr_inputs, -1)


