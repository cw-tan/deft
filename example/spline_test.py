import numpy as np
from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../build/'))
import pydeft as deft

# Gaussian in Fourier space
def f_tilde(kx,ky,kz):
  w = 0.2
  return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))

# define grid and box, and construct deft objects
box_vectors = 10*np.eye(3)
box = deft.Box(box_vectors)

print('{:^10} {:^16} {:^16} {:^16} {:^16} {:^16} {:^16}'.format('N', 'Naive', 'Spline-4', 'Spline-8', 'Spline-12', 'Spline-16', 'Spline-20'))

for N in np.linspace(10, 100, 10)**3:
  Ni = round(N**(1/3))
  shape = (Ni, Ni, Ni)
  x = np.linspace(0, 10, Ni)
  x,y,z = np.meshgrid(x,x,x, indexing = 'ij')
  
  points_array = np.empty((3, Ni, Ni, Ni))
  points_array[0,:,:,:] = x
  points_array[1,:,:,:] = y
  points_array[2,:,:,:] = z
  points_array = points_array.reshape(3, Ni*Ni*Ni).T
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde)
  end = timer()
  naive_time = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 4)
  end = timer()
  spline_time_4 = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 8)
  end = timer()
  spline_time_8 = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 12)
  end = timer()
  spline_time_12 = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 16)
  end = timer()
  spline_time_16 = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 20)
  end = timer()
  spline_time_20 = end - start
  
  print('{:^10} {:^16.4f} {:^16.4f} {:^16.4f} {:^16.4f} {:^16.4f} {:^16.4f}'\
       .format(Ni**3, naive_time, spline_time_4, spline_time_8, spline_time_12, spline_time_16, spline_time_20))

