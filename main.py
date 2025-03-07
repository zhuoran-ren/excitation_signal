"""This script is used to generate excitation signals
with random phases for system identification in the 
frequency domain.
"""
from src.excitation_signal import ExcitationSignal
from src.utils import *
from src.visualization import Visualization

def main(freq_range: tuple,
         f: float, 
         N: int,
         p: int,
         m: int,
         nr_inputs: int,
         mode: str,
         file_name: str,
         is_visualization: bool,
         is_save: bool) -> None:
    """Generate the multisine signals. Visualize and
    save the generated signals.

    Args:
        freq_range (tuple): The range of excited frequency.
        f (float): The sampling frequency.
        N (int): The number of sampled points.
        p (int): The number of repeat times of one signal.
        m (int): The number of different signals.
        nr_inputs (int): The number of inputs of the system.
        mode (str): The mode to generate signals for different input channels.
        file_name (str): The name of the signal file.
        is_visualization (bool): Whether to visualize the generated signals.
        is_save (bool): Whether to save the generated signals.
    """
    print_info(Range=(freq_range,),
               Frequency=(f,),
               Repeats=(p,),
               Types=(m,),
               Inputs=(nr_inputs,))
    
    generator = ExcitationSignal(freq_range=freq_range, f=f, N=N)
    U_amp, U_phase, u, us = generator.get_multi_signals(m=m, 
                                                        p=p, 
                                                        nr_inputs=nr_inputs,
                                                        mode=mode)
    t_stamp = generator.get_time_stamp(f, N*p*m)

    data = {
        'freq_range': freq_range,      # the excited frequency range
        'nr_inputs': nr_inputs,        # the number of inputs
        'mode': mode,                  # the way to generate signals for multiple inputs
        'f': f,                        # the sampling frequency
        'N': N,                        # the number of sampling points
        'p': p,                        # the repeat times of each signal
        'm': m,                        # the number of different signals
        'amplitude': U_amp,            # the amplitude in the frequency domain for each signal
        'phase': U_phase,              # the phase in the frequency domain for each signal
        'u': u,                        # the m different time signals for each input channel
        'us': us,                      # the m*p signals for each input channel
        'idx': generator.idx,          # the start and end index in the frequency stamp  
        'f_stamp': generator.f_stamp,  # the frequency stamp for each signal
        't_stamp': generator.t_stamp,  # time stamp for one signal
        't_stamps': t_stamp            # time stamp for one experiment
    }

    if is_save is True:
        save_file(data, file_name)
    
    if is_visualization is True:
        vis = Visualization(freq_range=freq_range,
                            U_amp=U_amp,
                            U_phase=U_phase,
                            u=u,
                            t_stamp=generator.t_stamp,
                            f_stamp=generator.f_stamp)
        vis.plot_signals()

if __name__ == '__main__':
    main(freq_range = (0.0, 4.0),
         f = 100.0,
         N = 1000,
         p = 5,
         m = 3,
         nr_inputs = 3,
         mode = 'orthogonal',
         file_name = 'test',
         is_visualization = True,
         is_save = True)