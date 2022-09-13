import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'hspace': 0})

time = np.loadtxt('spline.times', skiprows=1).T

axs[0].plot(time[0], time[1] / 60, 'kD', label='Naive')
axs[0].plot(time[0], time[2] / 60, 'ko', label='PME-4')
axs[0].plot(time[0], time[3] / 60, 'kx', label='PME-12')
axs[0].plot(time[0], time[4] / 60, 'k+', label='PME-20')

Nis = np.array([2, 5, 10, 22, 46])

errs4 = []
errs12 = []
errs20 = []

for Ni in Nis:
    naive = np.load('lattice_sum_benchmark/naive-{}.npy'.format(Ni))
    pme4 = np.load('lattice_sum_benchmark/pme4-{}.npy'.format(Ni))
    pme12 = np.load('lattice_sum_benchmark/pme12-{}.npy'.format(Ni))
    pme20 = np.load('lattice_sum_benchmark/pme20-{}.npy'.format(Ni))

    errs4.append(np.abs(naive-pme4).max() / np.ptp(naive) * 100)
    errs12.append(np.abs(naive-pme12).max() / np.ptp(naive) * 100)
    errs20.append(np.abs(naive-pme20).max() / np.ptp(naive) * 100)

axs[1].plot(Nis**3, errs4, '-ko', alpha=0.4)
axs[1].plot(Nis**3, errs12, '--kx', alpha=0.7, markersize=10)
axs[1].plot(Nis**3, errs20, '-.k+', alpha=1, markersize=10)

[axs[i].set_xscale('log') for i in range(2)]
[axs[i].set_yscale('log') for i in range(2)]

plt.show()

