import numpy as np
from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../build/'))
import pydeft as deft

# Gaussian in Fourier space
def f_tilde(kx,ky,kz):
  w = 1
  return np.sqrt(np.pi) * w * np.exp(-(kx*kx+ky*ky+kz*kz) *w*w / 4)

print('{:^10} {:^18} {:^18} {:^18} {:^18}'.format('N', 'Naive', 'Spline-4', 'Spline-12', 'Spline-20'))

Nis = [2, 5, 10, 22, 46]

for Ni in Nis:

  box = deft.Box(Ni * np.eye(3))
  shape = (4*Ni, 4*Ni, 4*Ni)
  
  x = np.linspace(0, Ni, Ni, endpoint = False)
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
  np.save('lattice_sum_benchmark/naive-{}'.format(Ni), grd)

  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 4)
  end = timer()
  spline_time_4 = end - start 
  np.save('lattice_sum_benchmark/pme4-{}'.format(Ni), grd)

  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 12)
  end = timer()
  spline_time_12 = end - start
  np.save('lattice_sum_benchmark/pme12-{}'.format(Ni), grd)

  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 20)
  end = timer()
  spline_time_20 = end - start
  np.save('lattice_sum_benchmark/pme20-{}'.format(Ni), grd)


  print('{:^10} {:^18.6f} {:^18.6f} {:^18.6f} {:^18.6f}'\
       .format(Ni**3, naive_time, spline_time_4, spline_time_12, spline_time_20))


