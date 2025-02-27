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
                 f: float=100.0, N: int=100, amp: int=100.0,
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
        self.initialization(self.freq_range, self.f, self.N)
        self.amp = amp
        self.eps = eps

    def initialization(self, freq_range: tuple,
                       f: float, N: int) -> None:
        """Initialize the necessary parameters.

        Args:
            freq_range (tuple): The excited frequency range.
            f (float): The sampling frequency.
            N (int): The number of points of each signal.
        
        Returns:
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

    def _get_frequency_signal(self, N: int, amp: float, 
                              idx: tuple) -> complex:
        """Get one frequency signal.
        """
        U_amp = self._get_frequency_amplitude(N, amp, idx)
        U_phase = self._get_random_phase(N)
        U = self._get_complex_signal(U_amp, U_phase)
        return U, U_amp, U_phase

    @staticmethod
    def _get_time_signal(U: complex):
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

    def _get_excitation_signal(self, N: int,
                               amp: float,
                               idx: tuple) -> tuple[np.ndarray, 
                                                    np.ndarray,
                                                    np.ndarray]:
        """Generate one excitation signal with random
        phase in the frequency domain.

        Args:
            N (int): The number of points of a signal.
            amp (float): The amplitude in the frequency domain.
            idx (tuple): The start and end indices of the excited range. 

        Returns:
            U (complex): The frequency signal.
            U_amp (array): The amplitude in the frequency domain.
            U_phase (array): The phase in the frequency domain.
            u (array): The time signal. 
        """
        U, U_amp, U_phase = self._get_frequency_signal(N, amp, idx)
        u = self._get_time_signal(U)
        norm_u = self._get_normalization(u)
        return U_amp, U_phase, norm_u

    def get_signal(self, p: int, 
                   N: int, 
                   amp: int, 
                   idx: tuple) -> tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
        """Get one signal repeating p times. Ensuring that the 
        difference between the beginning and the end of the same 
        signal is not too large.
        
        Args:
            p (int): The number of repeat times.

        Returns:
            U_amp (array): The amplitude in the frequency domain.
            U_phase (array): The random phase in the frequency domain.
            u (array): The single time signal.
            us (array): The repeated time signal.
        """
        diff = 1000.0
        while diff > self.eps:
            U_amp, U_phase, u = self._get_excitation_signal(N, amp, idx)
            diff = np.abs(u[0] - u[1])
        return U_amp, U_phase, u, np.tile(u, p)
    
    def get_multi_signals(self, m: int, 
                          p: int) -> tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           np.ndarray]:
        """Generate m different signals, and each signal repeats
        p times.

        Args:
            m (int): The number of different signals.
            p (int): Each signal repeats p times.
        
        Returns:
            U_amp (m x N): The amplitude in the frequency domain of different signals.
            U_Phase (m x N): The random phase in the frequency domain of differnt signals.
            u (m x N): The different signals in the time domain.
            us (m x N*p): The m different signals repeated p times.
        """
        U_amp = np.zeros((m, self.N))
        U_phase = np.zeros((m, self.N))
        u = np.zeros((m, self.N))
        us = np.zeros((m, p*self.N))

        for i in tqdm(range(m)):
            U_amp[i, :], U_phase[i, :], u[i, :], us[i, :] = self.get_signal(p, self.N, self.amp, self.idx)
        
        return U_amp, U_phase, u, us
    
