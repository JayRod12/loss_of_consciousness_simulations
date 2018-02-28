import scipy.io as spio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']

xs, ys, zs = XYZ.transpose()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)

plt.show()
