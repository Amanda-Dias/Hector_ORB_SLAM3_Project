import numpy as np
from matplotlib import pyplot as plt
import similaritymeasures
import rosbag
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.interpolate import interp1d
import random

#Bag
bag = rosbag.Bag('hector_orbslam3_straightline.bag')

old_hector=[]
old_odom=[]
old_orb=[]

hector= []
odom= []
orb_slam= []
origem= []


def proportion(odom,orb,xy=None):
    
    min_x_A = min(x for x, y in orb)
    max_x_A = max(x for x, y in orb)
    min_y_A = min(y for x, y in orb)
    max_y_A = max(y for x, y in orb)

    min_x_B = min(x for x, y in odom)
    max_x_B = max(x for x, y in odom)
    min_y_B = min(y for x, y in odom)
    max_y_B = max(y for x, y in odom)

    scale_factor_x = (max_x_B - min_x_B) / (max_x_A - min_x_A)
    scale_factor_y = (max_y_B - min_y_B) / (max_y_A - min_y_A)
    return scale_factor_x, scale_factor_y


for topic, msg, t in bag.read_messages(topics=['/slam_out_pose','/odom','/orb_slam3/camera_pose']):
    
    if topic == '/slam_out_pose':
        
        hector.append((msg.pose.position.x, msg.pose.position.y))
        
    elif topic == '/odom':
        
        odom.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
     
    elif topic == '/orb_slam3/camera_pose':
        scale_axle_x_x=1
        scale_axle_x_y=1
        scale_axle_x_z=1
        x1=(1*scale_axle_x_x*msg.pose.position.z)
        y1=(-1*scale_axle_x_y*msg.pose.position.x)
        z1=(1*scale_axle_x_z*msg.pose.position.y)
        orb_slam.append((x1,y1)) 

bag.close()

const_x, const_y= proportion(odom,orb_slam,0)
prop_orb = [(x * const_x, y * const_y) for x, y in orb_slam]

orb_slam_array = np.array(prop_orb)
hector_array = np.array(hector)
odom_array = np.array(odom)

fig, ax = plt.subplots()

# Zoom Configuration: configures the zoom region in the image using an insertion graphic.

# Plot your trajectories as before
ax.plot(*zip(*odom_array), color='red', label='Odom (Ground Truth)', linestyle='-')
ax.scatter(*zip(*hector_array), edgecolors='green', facecolors='none', label=f'Hector SLAM', marker='s')
ax.scatter(*zip(*orb_slam_array), edgecolors='blue', facecolors='none', label=f'ORB-SLAM3', marker='o')

ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.legend()

# Manually define the ranges and scales of the x and y axes

ax.set_xlim(-1.5,1.5)  # Range and scale for x-axis
ax.set_ylim(-1.5,1.5)  # Range and scale for the y-axis

x1, x2, y1, y2 = 0.8, 0.95, -0.08, 0.00  # subregion of the original image

#ODOM 
odom_array_x = odom_array[:, 0]
odom_array_x_inzoom = (odom_array_x >= x1) & (odom_array_x <= x2)
odom_array_y = odom_array[:, 1]
odom_array_y_inzoom = (odom_array_y >= y1) & (odom_array_y <= y2)
odom_array_zoom = np.where((odom_array_x_inzoom & odom_array_y_inzoom))

#HECTOR
hector_array_x = hector_array[:, 0]
hector_array_x_inzoom = (hector_array_x >= x1) & (hector_array_x <= x2)
hector_array_y = hector_array[:, 1]
hector_array_y_inzoom = (hector_array_y >= y1) & (hector_array_y <= y2)
hector_array_zoom = np.where((hector_array_x_inzoom & hector_array_y_inzoom))

#ORB-SLAM3
orb_slam_array_x = orb_slam_array[:, 0]
orb_slam_array_x_inzoom = (orb_slam_array_x >= x1) & (orb_slam_array_x <= x2)
orb_slam_array_y = orb_slam_array[:, 1]
orb_slam_array_y_inzoom = (orb_slam_array_y >= y1) & (orb_slam_array_y <= y2)
orb_slam_array_zoom = np.where((orb_slam_array_x_inzoom & orb_slam_array_y_inzoom))


axins = ax.inset_axes(
    [0.3, 0.1, 0.3, 0.3],xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    

axins.plot(*zip(*odom_array[odom_array_zoom]), color='red', linestyle='-')
axins.plot(*zip(*hector_array[hector_array_zoom]), color='green',marker='s')
axins.plot(*zip(*orb_slam_array[orb_slam_array_zoom]), color='blue',marker='o')
ax.indicate_inset_zoom(axins, edgecolor="black")

# Show the figure
plt.grid()
plt.show()





    