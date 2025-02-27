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
        file_name (str): The name of the signal file.
        is_visualization (bool): Whether to visualize the generated signals.
        is_save (bool): Whether to save the generated signals.
    """
    print_info(Range=(freq_range,),
               Frequency=(f,),
               Repeats=(p,),
               Types=(m,))
    
    generator = ExcitationSignal(freq_range=freq_range, f=f, N=N)
    U_amp, U_phase, u, us = generator.get_multi_signals(m=m, p=p)
    t_stamp = generator.get_time_stamp(f, N*p*m)

    data = {
        'freq_range': freq_range,
        'f': f,
        'N': N,
        'p': p,
        'm': m,
        'amplitude': U_amp,
        'phase': U_phase,
        'u': u,
        'us': us,
        'f_stamp': generator.f_stamp,
        't_stamp': generator.t_stamp,
        't_stamps': t_stamp
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
        vis.plot_signals(nr=3)

if __name__ == '__main__':
    main(freq_range = (0.0, 10.0),
         f = 100.0,
         N = 1000,
         p = 10,
         m = 10,
         file_name = 'test',
         is_visualization = True,
         is_save = True)