import math
import typing as T

import numpy as np
from numpy import linalg
#from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    # matrix of basis functions 
    t0 = 0.0 # initial time 
    A = np.array([
        [1, t0, t0**2, t0**3],
        [0, 1, 2*t0, 3*t0**2],
        [1, tf, tf**2, tf**3],
        [0, 1, 2*tf, 3*tf**2]
    ])

    # initial and final states 
    b_x = np.array([initial_state.x, initial_state.xd, final_state.x, final_state.xd])
    b_y = np.array([initial_state.y, initial_state.yd, final_state.y, final_state.yd])
    
    coeffs_x = np.linalg.solve(A, b_x)
    coeffs_y = np.linalg.solve(A, b_y)

    # put x and y coefficients into one array     
    coeffs = np.concatenate([coeffs_x, coeffs_y])
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7))
    ########## Code starts here ##########

    # create matrices of bases 
    T_matrix = np.array([t**0, t**1, t**2, t**3]).T
    dT_matrix = np.array([0*t**0, 1*t**0, 2*t**1, 3*t**2]).T
    ddT_matrix = np.array([0*t, 0*t, 2+0*t, 6*t]).T

    # get the coefficient matrices of the flat outputs x and y 
    coeffs_x = coeffs[0:4]
    coeffs_y = coeffs[4:8]
    x   = T_matrix @ coeffs_x
    y   = T_matrix @ coeffs_y
    dx  = dT_matrix  @ coeffs_x
    dy  = dT_matrix  @ coeffs_y
    ddx = ddT_matrix @ coeffs_x
    ddy = ddT_matrix @ coeffs_y
    th = np.arctan2(dy, dx)
    
    # put everything together 
    traj = np.column_stack([x, y, th, dx, dy, ddx, ddy])
    ########## Code ends here ##########

    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    dx = traj[:,3]
    dy = traj[:,4]
    th = traj[:,2]
    ddx = traj[:,5]
    ddy = traj[:,6]
    V = np.sqrt(dx**2 + dy**2)
    # dv = np.sqrt(ddx**2 + ddy**2)
    # om = (ddx - dv * np.cos(th)) / (-1 * v * np.sin(th))
    a = ddx * np.cos(th) + ddy * np.sin(th)
    numerator = -ddx * np.sin(th) + ddy * np.cos(th)
    epsilon = 1e-6  # A small number to prevent division by zero
    om = numerator / (V + epsilon)
    ########## Code ends here ##########
    return V, om

if __name__ == "__main__":
    # Constants
    tf = 25.

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=0.5, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=0.5, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.savefig("plots/differential_flatness.png")
    plt.show()
