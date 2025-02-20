"""
All distances are in cm
For details on the model, refer to Meeting notes: Refraction model (Notability)
"""

import cv2 as cv
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import pickle
import matplotlib.pyplot as plt
from config import Config

def refraction_model(y0, prism_object_distance_):
    """
    y0: y coordinate of the point of intersection on the first face of the prism in the world reference frame
    prism_object_distance_: distance between the object and the prism's first face
    """
    y = prism_camera_distance * (object_point - y0) / prism_object_distance_ #y coordinate of the point of intersection on the second face of the prism in the world reference frame
    return np.abs(y / np.sqrt(prism_camera_distance ** 2 + y ** 2) - n_glass_air * (y0 - y) / np.sqrt(prism_thickness ** 2 + (y0 - y) ** 2))

def get_initial_guess(object_point, prism_object_distance_):
    min_error = 1e6
    for y0 in np.linspace(0, 50 * object_height, 100 * num_points):
        if np.abs(refraction_model(y0, prism_object_distance_)) < min_error:
            min_error = np.abs(refraction_model(y0, prism_object_distance_))
            best_y0 = y0
    return best_y0

def convert_world_to_pixel_coordinates(world_x, world_y, camera_rotation):
    """
    world_x: x coordinate of the point on the camera sensor in the world reference frame
    world_y: y coordinate of the point on the camera sensor in the world reference frame
    camera_rotation: angle of rotation of the camera from the world reference frame
    returns: distance of the point from the center of the camera sensor in the pixel reference frame
    """
    camera_center_x = -focal_length * np.cos(camera_rotation)
    camera_center_y = -focal_length * np.sin(camera_rotation)
    camera_projection_direction_unit_vector = np.array([0, 1])
    rot_mat = np.array([[np.cos(camera_rotation), -np.sin(camera_rotation)], [np.sin(camera_rotation), np.cos(camera_rotation)]])
    camera_projection_direction_unit_vector = rot_mat @ camera_projection_direction_unit_vector
    image_projection_vector = np.array([world_x[0], world_y[0]]) - np.array([camera_center_x, camera_center_y])
    return pixel_scaling_factor * (image_projection_vector @ camera_projection_direction_unit_vector)
    
def find_intersection_with_camera_sensor(focal_length_, camera_rotation_, line_slope_):
    """
    focal_length_: focal length of the camera (pin-hole model)
    camera_rotation_: angle of rotation of the camera from the world reference frame
    line_slope_: slope of the line joining the camera center and the camera sensor
    """
    camera_intersection_x = -focal_length_ / (np.cos(camera_rotation_) + np.sin(camera_rotation_) * line_slope_) 
    camera_intersection_y = camera_intersection_x * line_slope
    return camera_intersection_x, camera_intersection_y

def get_slope_of_refracted_line(prism_camera_distance_, object_point_, y0_solution_, prism_object_distance_):
    y = prism_camera_distance_ * (object_point_ - y0_solution_) / prism_object_distance_
    line_slope_ = y / prism_camera_distance_
    return line_slope_

num_points = Config.num_points
refractive_index_glass = Config.refractive_index_glass # N-BK7 glasss
refractive_index_air = Config.refractive_index_air
n_glass_air = refractive_index_glass / refractive_index_air
prism_object_distance_config = Config.prism_object_distance
prism_camera_distance = Config.prism_camera_distance
prism_thickness = Config.prism_thickness
focal_length = Config.focal_length
object_camera_distance = prism_object_distance_config + prism_thickness + prism_camera_distance
object_height = Config.object_height
pixel_scaling_factor = 1
camera_rotation = Config.theta_camera # rotated downwards is positive

with open('/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/data/simulation_coordinates.pkl', 'rb') as f:
    data = pickle.load(f)

object_points = np.linspace(-object_height, object_height, num_points)
prism_object_distances = np.linspace(0.6 * prism_object_distance_config, 1.6 * prism_object_distance_config, num_points)
#object_points = data[1][:,1]
y0_kristins_simulation = data[2][:,1]
image_points_kristins_simulation_world = data[0]
image_points_kristins_simulation_pixels = np.linspace(-Config.image_height, 0, num_points) * pixel_scaling_factor
image_points_pixels = np.zeros(len(object_points))
image_points_world = np.zeros((len(object_points), 2))

for point_id, (object_point, prism_object_distance) in enumerate(zip(object_points, prism_object_distances)):
    y0_initial_guess = get_initial_guess(object_point, prism_object_distance)
    y0_solution = least_squares(refraction_model, y0_initial_guess, args=(prism_object_distance,)).x   
    line_slope = get_slope_of_refracted_line(prism_camera_distance, object_point, y0_solution, prism_object_distance)
    camera_intersection_x, camera_intersection_y = find_intersection_with_camera_sensor(focal_length, camera_rotation, line_slope)
    image_points_world[point_id,:] = np.array([camera_intersection_x, camera_intersection_y])[:,0]
    image_points_pixels[point_id] = convert_world_to_pixel_coordinates(camera_intersection_x, camera_intersection_y, camera_rotation)

fig, ax = plt.subplots(3,1,figsize=(13,20))
colors = plt.get_cmap('jet')(np.linspace(0, 1.0, num_points))
for i in range(num_points):
    ax[0].plot(i, image_points_pixels[i],'o',color=colors[i])
ax[0].plot(image_points_pixels, color='black')
ax[0].set_xlabel('Image points id', fontsize = 24)
ax[0].set_ylabel('Image points distance', fontsize=24)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].tick_params(axis='y', labelsize=18)

for i in range(num_points):
    ax[1].plot(i, object_points[i],'o',color=colors[i])
ax[1].plot(object_points, color='black')
ax[1].set_xlabel('Object points id', fontsize = 24)
ax[1].set_ylabel('Object points distance', fontsize = 24)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)

for i in range(num_points):
    ax[2].plot(image_points_pixels[i], object_points[i],'o',color=colors[i])
ax[2].plot(image_points_pixels, object_points, color='black')
ax[2].set_xlabel('Image points distance', fontsize = 24)
ax[2].set_ylabel('Object points distance', fontsize = 24)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].tick_params(axis='y', labelsize=18)
plt.savefig('fig1.png')
plt.axis('equal')

plt.figure(figsize=(25,25))
for i in range(num_points):
    plt.scatter(image_points_world[i,0], image_points_world[i,1], s=8, marker='o', color=colors[i])
    plt.scatter(prism_object_distances[i], object_points[i], s=35, marker='x', color=colors[i])
plt.xlabel('x', fontsize=34)
plt.ylabel('y', fontsize=34)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(['Image Points', 'Object Points'], fontsize=24)
plt.savefig('fig2.png')
plt.axis('equal')

plt.figure(figsize=(25,25))
for i in range(num_points):
    plt.plot(image_points_world[i,0], image_points_world[i,1], 'o', color=colors[i], markersize=4)
    plt.plot(image_points_kristins_simulation_world[i,0], image_points_kristins_simulation_world[i,1],
     'x', color=colors[i], markersize=4)
plt.xlabel('x', fontsize=24)
plt.ylabel('y', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['Forward', 'Backward'])
plt.axis('equal')
#plt.savefig('fig2.png')
