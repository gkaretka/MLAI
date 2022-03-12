from pylfsr import LFSR
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from LSFR.sum_elements import sum_elements
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.api import plot_corr

from scipy.signal import correlate

"""
1 - 4
0 - 4
11 - 2
00 - 2
111 - 1
000 - 1
11111 - 1
0000 - 1
"""
L1 = LFSR(fpoly=[5, 4, 3, 2], initstate=[0, 0, 0, 1, 1]) #[0, 1, 0, 1, 1], #[1, 0, 0, 1, 1]
L2 = LFSR(fpoly=[5, 2], initstate=[1, 0, 0, 0, 0])

threshold = False
upsample = False
use_mtpltlib = True
use_scipy = True
upsample_cnt = 1

arr2 = []
arr1 = []
for i in range(32):
    L1.next()
    L2.next()
    arr1.append(int(L1.outbit))
    arr2.append(int(L2.outbit))

if upsample:
    arr1 = np.array(arr1).repeat(upsample_cnt, axis=0)
    arr2 = np.array(arr2).repeat(upsample_cnt, axis=0)
else:
    arr1 = np.array(arr1, dtype="float32")
    arr2 = np.array(arr2, dtype="float32")


if use_scipy:
    res = correlate(arr1, arr2)
    sc1 = correlate(arr1, arr1, mode="same")
    sc2 = correlate(arr2, arr2)
else:
    res = sm.tsa.stattools.ccf(arr1, arr2, adjusted=True)
    sc1 = sm.tsa.acf(arr1)
    sc2 = sm.tsa.acf(arr2)

x = [len(res)/2 - (1+i) for i in range(len(res))]
x_ac1 = [len(sc1)/2 - (1+i) for i in range(len(sc1))]
x_ac2 = [len(sc2)/2 - (1+i) for i in range(len(sc2))]

fig: plt.subplots
fig, ax = plt.subplots(1, 3, figsize=(3, 1))

# add grid to all plots
for i in range(len(ax)):
    ax[i].grid(which='both', linestyle='--')

# threshold
if threshold:
    sc1 = [0 if elm < 0.3 else 1 for elm in sc1]
    sc2 = [0 if elm < 0.3 else 1 for elm in sc2]

# x_ac1, x_ac2, x
ax[0].scatter(list(range(len(sc1))), sc1)
ax[1].scatter(list(range(len(sc2))), sc2)

if use_mtpltlib:
    ax[2].xcorr(arr1, arr2, usevlines=True, maxlags=15, normed=True)
else:
    ax[2].scatter(list(range(len(res))), res)

ax[0].set_ylabel('Value [-]')
ax[0].set_xlabel('N [-]')
ax[1].set_xlabel('N [-]')
ax[2].set_xlabel('N [-]')

ax[0].set_title("Autocorelation PN 1")
ax[1].set_title("Autocorelation PN 2")
ax[2].set_title("Cross-correlation PN 1 and PN 2")

merged = np.concatenate((np.array(sc1), np.array(sc2)), axis=0)

ymax, ymin = merged.max(initial=0, axis=0), merged.min(initial=10000, axis=0)
ax[0].set_ylim(ymin, ymax)
ax[1].set_ylim(ymin, ymax)

if not use_mtpltlib:
    ymax, ymin = res.max(initial=0, axis=0), res.min(initial=10000, axis=0)
    ax[2].set_ylim(ymin, ymax)

plot_acf(sc1)
plot_acf(sc2)

m2 = np.concatenate((np.array([sc1]), np.array([sc2])), axis=1)

plot_corr(m2)

plt.show()
