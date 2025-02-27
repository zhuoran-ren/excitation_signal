"""Visualize the saved data from files.
"""
import numpy as np

from src.utils import *
from src.visualization import Visualization


def load_data(file_name) -> tuple[tuple,
                                  np.ndarray,
                                  np.ndarray,
                                  np.ndarray,
                                  np.ndarray,
                                  np.ndarray]:
    """Load the data from files.
    """
    data = load_file(file_name)
    return (data['freq_range'],
            data['amplitude'],
            data['phase'],
            data['u'],
            data['t_stamp'],
            data['f_stamp'])

def main(file_name: str) -> None:
    """Load the data from file and visualize it.
    """
    (freq_range, U_amp, U_phase, u, t_stamp, f_stamp) = load_data(file_name)
    vis = Visualization(freq_range=freq_range,
                            U_amp=U_amp,
                            U_phase=U_phase,
                            u=u,
                            t_stamp=t_stamp,
                            f_stamp=f_stamp)
    vis.plot_signals(nr=3)
    
if __name__ == '__main__':
    file_name = 'test'
    main(file_name)