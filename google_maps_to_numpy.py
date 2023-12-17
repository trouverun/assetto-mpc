import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas
from scipy import interpolate
import pymap3d as pm

matplotlib.use("qtagg")

json_track = pandas.read_json("tracks/rlywide.json")
midpoints = json_track.get("midpoint")
m1 = json_track.get("marker1")
m2 = json_track.get("marker2")

car_pos = [pm.geodetic2enu(
    midpoints[i]["lat"], midpoints[i]["lng"], 0,
    midpoints[8]["lat"], midpoints[8]["lng"], 0
) for i in range(len(midpoints))]
numpy_track = np.asarray(car_pos)

m1 = [pm.geodetic2enu(
    m1[i]["lat"], m1[i]["lng"], 0,
    midpoints[8]["lat"], midpoints[8]["lng"], 0
) for i in range(len(m1))]
m1 = np.asarray(m1)

m2 = [pm.geodetic2enu(
    m2[i]["lat"], m2[i]["lng"], 0,
    midpoints[8]["lat"], midpoints[8]["lng"], 0
) for i in range(len(m2))]
m2 = np.asarray(m2)


angle = np.deg2rad(90.375)
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
angle2 = np.deg2rad(0.015)
R2 = np.array([
    [np.cos(angle2), -np.sin(angle2)],
    [np.sin(angle2), np.cos(angle2)]
])

old_shift = [-159, +104, 0]
shift = [-10.5, -27, 0]

# traj = np.load("trajectory_wrong.npy")
# traj[:, 1:4] += (np.asarray(shift) - old_shift)
# np.save("trajectory.npy", traj)

numpy_track[:, :2] = (R @ numpy_track[:, :2].T).T
numpy_track += shift
m1[:, :2] = (R @ m1[:, :2].T).T
m1 += shift
m2[:, :2] = (R @ m2[:, :2].T).T
m2 += shift


numpy_track[:, :2] = (R2 @ numpy_track[:, :2].T).T
m1[:, :2] = (R2 @ m1[:, :2].T).T
m2[:, :2] = (R2 @ m2[:, :2].T).T

widths = np.sqrt(np.sum(np.square(m1[:, :2] - m2[:, :2]), axis=1)) / 2 - 1.2

distances = np.sqrt(np.sum(np.square(numpy_track[1:, :2] - numpy_track[:-1, :2]), axis=1))
distances = np.r_[0, np.cumsum(distances)]
print(distances.shape)
print(widths.shape)
numpy_track = np.c_[distances, numpy_track[:, :2] * [1, 1], widths]
print(numpy_track.shape)

np.save("tracks/spain_gps.npy", numpy_track)

cx_spline = interpolate.splrep(numpy_track[:, 0], numpy_track[:, 1], k=3)
cy_spline = interpolate.splrep(numpy_track[:, 0], numpy_track[:, 2], k=3)
cw_spline = interpolate.splrep(numpy_track[:, 0], numpy_track[:, 3], k=3)
inters = np.linspace(0, numpy_track[-1, 0], 1000)
x = interpolate.splev(inters, cx_spline, der=0)
y = interpolate.splev(inters, cy_spline, der=0)
dx = interpolate.splev(inters, cx_spline, der=1)
dy = interpolate.splev(inters, cy_spline, der=1)
hdg = np.arctan2(dy, dx)
w = interpolate.splev(inters, cw_spline, der=0)

m1x = x + np.cos(hdg+np.pi/2) * w
m1y = y + np.sin(hdg+np.pi/2) * w
m2x = x + np.cos(hdg-np.pi/2) * w
m2y = y + np.sin(hdg-np.pi/2) * w

manual_track = np.load("tracks/manual_spain.npy")

# angle = np.deg2rad(-124.5)
# R = np.array([
#     [np.cos(angle), -np.sin(angle)],
#     [np.sin(angle), np.cos(angle)]
# ])
# manual_track = (R @ manual_track[:, 1:3].T).T

plt.figure(figsize=(20, 10))
plt.scatter(manual_track[:, 1], manual_track[:, 2], color="r")
# plt.scatter(x, y, color="b")
plt.scatter(m1x, m1y, color="g")
plt.scatter(m2x, m2y, color="g")
plt.show()
