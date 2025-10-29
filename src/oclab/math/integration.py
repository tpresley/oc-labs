import numpy as np
from typing import Callable

def rk4_step(y, t, dt, f: Callable):
    k1 = f(t, y)
    k2 = f(t+0.5*dt, y+0.5*dt*k1)
    k3 = f(t+0.5*dt, y+0.5*dt*k2)
    k4 = f(t+dt, y+dt*k3)
    return y + (dt/6.0)*(k1+2*k2+2*k3+k4)
