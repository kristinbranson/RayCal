import numpy as np

class Config:
    num_points = 25
    refractive_index_glass = 1.516 # N-BK7 glasss
    refractive_index_air = 1.0003
    #refractive_index_glass = refractive_index_air
    prism_object_distance = 1
    prism_camera_distance = 15
    prism_thickness = 3
    focal_length = 2.5
    image_height = 1
    theta_camera = np.pi / 6 # rotated downwards is positive
    object_height = 1
