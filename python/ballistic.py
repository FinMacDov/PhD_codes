import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg') # revert above
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

def ballistic_flight(v0, g, t):
    # assumes perfectly verticle launch and are matching units
    # v0-initial velocity
    # g-gravitational acceleration
    # t-np time array
    x = v0*t
    y = v0*t-0.5*g*t**2
    y = np.where(y<0,0,y)

    t_apex = v0/g
    x_apex = v0*t_apex    
    y_apex = v0*t_apex-0.5*g*(t_apex)**2
    return x, y, t_apex, x_apex, y_apex

m_2_km = 1e-3
v0 = 6e4*m_2_km  #km s-1

earth_g = 9.80665  #m s-2
sun_g = 28.02*earth_g*m_2_km # km s-2

t = np.linspace(0,500,1000)

test = ballistic_flight(v0, sun_g, t)

plt.plot(t,test[1])
