from fastdtw import fastdtw
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
from scipy.spatial.distance import directed_hausdorff


#Bag
bag = rosbag.Bag('hector_orbslam3_straightline.bag')
old_hector=[]
old_odom=[]
old_orb=[]

hector=[]
odom= []
orb_slam=[]


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
        scale_axle_x=1
        scale_axle_y=1
        scale_axle_z=1
        x1=(1*scale_axle_y*msg.pose.position.z)
        y1=(-1*scale_axle_x*msg.pose.position.x)
        z1=(1*scale_axle_z*msg.pose.position.y)
        orb_slam.append((x1,y1)) 
        

bag.close()

const_x, const_y= proportion(odom,orb_slam,0)
prop_orb = [(x * const_x, y * const_y) for x, y in orb_slam]

hector_array = np.array(hector)
odom_array = np.array(odom)
orb_slam_array = np.array(prop_orb)

# Calculate the Hausdorff Distance between the sets
distance_orbslam_to_odom = directed_hausdorff(orb_slam_array, odom_array)[0]
distance_hector_to_odom = directed_hausdorff(hector_array , odom_array)[0]

# Calculate the Frechet Distance between the sets
frechet_distance_hector_odom = similaritymeasures.frechet_dist(hector_array, odom_array)
frechet_distance_orbslam_odom = similaritymeasures.frechet_dist(orb_slam_array, odom_array)

# Calculate the PCM Method between sets
pcm_hector_odom = similaritymeasures.pcm(hector_array, odom_array)
pcm_orbslam_odom = similaritymeasures.pcm(orb_slam_array, odom_array)

# Print the results
print("Hector FD Error for ODOM: ",frechet_distance_hector_odom)
print("ORB SLAM3 FD Error for ODOM: ",frechet_distance_orbslam_odom)

print("Hector DH Error for ODOM:", distance_hector_to_odom)
print("ORB SLAM3 DH Error for ODOM:", distance_orbslam_to_odom)

print("Hector PCM Error for ODOM:",pcm_hector_odom)
print("ORB SLAM3 PCM Error for ODOM:",pcm_orbslam_odom)


