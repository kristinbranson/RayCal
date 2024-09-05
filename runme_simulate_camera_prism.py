import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from config import Config
from scipy.spatial.transform import Rotation as R
mpl.use('tkAgg')
plt.ion()

def rotx(angle):
    """
    Rotation matrix around x-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (np.array): Rotation matrix.
    """
    rotation = np.array([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
    return rotation

def roty(angle):
    """
    Rotation matrix around y-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (np.array): Rotation matrix.
    """
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
    return rotation

def rotz(angle):
    """
    Rotation matrix around z-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (np.array): Rotation matrix.
    """
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
    return rotation

# %%Ray class (for a ray of light)
class light_ray():
    def __init__(self, origin=[0,0,0], direction=[1,0,0]):
        """
        Parameters:
        - origin (2-D list): Origin of the ray.
        - direction (2-D list): Direction of the ray.
        """
        if not isinstance(origin, np.ndarray):
            origin = np.array(origin)
        if not isinstance(direction, np.ndarray):
            direction = np.array(direction)

        self.origin = origin
        self.direction = direction
    
    def build_ray(self, point1, point2):
        """
        Build a ray from two points.
        Parameters:
        - point1 (2-D list): First point (this will become the origin of the ray).
        - point2 (2-D list): Second point.
        """
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        self.origin = point1
        self.direction = point2 - point1

    def visualize(self, fig=None, ax=None):
        """
        Visualize the ray.
        """
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.origin[0], self.origin[1], self.origin[2], marker='o')
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], self.direction[0], self.direction[1], self.direction[2])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()
        return fig, ax


# %% Plane class
class Plane():
    def __init__(self, normal=[1,0,0], center=[0,0,0], alpha=0., beta=0., gamma=0., a=1., b=1.):
        """
        NOTE: Default normal vector is [1, 0, 0], and center is [0,0,0]
        Parameters:
        - normal (2-D list): Normal of the plane.
        - center (2-D list): Center of the plane.
        - alpha (float): Angle of the plane with respect to the x-axis.
        - beta (float): Angle of the plane with respect to the y-axis.
        - gamma (float): Angle of the plane with respect to the z-axis.
        - a (float): Width of the plane (for the default plane, along Y-axis).
        - b (float): Height of the plane (for the default plane, along Z-axis).
        """
        if not isinstance(normal, np.ndarray):
            normal = np.array(normal)
        if not isinstance(center, np.ndarray):
            center = np.array(center)

        if len(normal.shape) == 1:
            normal = normal.reshape((3, 1))
        
        if len(center.shape) == 1:
            center = center.reshape((3, 1))
        
        self.normal = normal
        self.center = center
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.update_sides()
        self.reposition()

    def reposition(self, alpha=None, beta=None, gamma=None, center=None):
        """
        Orient the plane.
        Get the equivalent angles of the resulting plane.
        NOTE: alpha, beta, gamma are always defined assuming the default normal vector is [1, 0, 0].
        Parameters:
        - alpha (float): Angle of the plane with respect to the x-axis.
        - beta (float): Angle of the plane with respect to the y-axis.
        - gamma (float): Angle of the plane with respect to the z-axis.
        """
        if alpha is None:
            alpha = self.alpha
        else:
            self.alpha = alpha
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
        if gamma is None:
            gamma = self.gamma
        else:
            self.gamma = gamma
        if center is None:
            center = self.center
        else:
            if not isinstance(center, np.ndarray):
                center = np.array(center)
            if len(center.shape == 1):
                center = center.reshape((3, 1))
            self.center = center
        
        normal = self.normal
        Rx = rotx(self.alpha)
        Ry = roty(self.beta)
        Rz = rotz(self.gamma)
        self.normal = Rz @ Ry @ Rx @ normal
        self.alpha, self.beta, self.gamma = self.get_angles(self.normal)

    def get_intersection(self, ray):
        """
        Get the intersection point of a ray with the plane.
        Parameters:
        - ray (np.array): Ray to intersect with the plane.
        Returns:
        - intersection (np.array): Intersection point.
        """
        ray_direction = ray.direction
        ray_origin = ray.origin
        t = np.dot(self.center - ray_origin, self.normal) / np.dot(ray_direction, self.normal)
        intersection = ray_origin + t * ray_direction
        return intersection

    def update_sides(self, side1=None, side2=None, side3=None, side4=None, alpha=None, beta=None, gamma=None):
        """
        Get the sides of the plane.
        Returns:
        - sides (2-D list): Four sides of the plane.
        Order of the sides: side parallel to Y-axis and in the positive Z region
                            side parallel to Y-axis and in the negative Z region
                            side parallel to Z-axis and in the positive Y region
                            side parallel to Z-axis and in the negative Y region
        """
        if side1 is None:
            side1 = np.array([[0., self.a/2, self.b/2], [0., -self.a / 2, self.b / 2]]).T - self.center
        else:
            side1 = self.side1 - self.center
        if side2 is None:
            side2 = np.array([[0., self.a/2, -self.b/2], [0., -self.a / 2, -self.b / 2]]).T - self.center
        else:
            side2 = self.side2 - self.center
        if side3 is None:
            side3 = np.array([[0., self.a/2, self.b/2], [0., self.a/2, -self.b/2]]).T - self.center
        else:
            side3 = self.side3 - self.center
        if side4 is None:
            side4 = np.array([[0., -self.a/2, self.b/2], [0., -self.a/2, -self.b/2]]).T - self.center
        else:
            side4 = self.side4 - self.center

        rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
        side1 = self.center + rot_mat @ side1
        side2 = self.center + rot_mat @ side2
        side3 = self.center + rot_mat @ side3
        side4 = self.center + rot_mat @ side4
        
        self.sides =  [side1, side2, side3, side4]

    def get_angles(self, norma=None):
        """
        Get the angles of the plane with respect to the x, y, and z axes.
        Returns:
        - angles (2-D list): Angles of the plane with respect to the x, y, and z axes.
        """
        self.update_sides()
        s10 = self.sides[0][:,0][:, None]
        print(norma)
        if norma is None:
            norma = self.normal
        if not isinstance(norma, np.ndarray):
            norma = np.array(norma)
        if len(norma.shape) == 1:
            norma = norma.reshape((3, 1))
        # Make sure the normal vector is normalized
        norma = norma / np.linalg.norm(norma)
        gamma = np.arctan2(norma[1], norma[0])[0]
        print(gamma)
        norma = rotz(-gamma) @ norma
        s10 = rotz(-gamma) @ s10
        beta = np.arctan2(-norma[2], norma[0])[0]
        print(beta)
        norma = roty(-beta) @ norma
        s10 = roty(-beta) @ s10
        print(s10)
        assert np.abs(s10[0]) < 1e-5
        alpha = np.arctan2(s10[2], s10[1])[0] - np.arctan2(self.b, self.a)
        return [alpha, beta, gamma]

    def visualize(self, fig=None, ax=None):
        """
        Visualize the plane by plotting the sides and 500 points lying on the plane.
        """
        rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
        sampled_points = np.random.rand(3, 500)
        sampled_points[0,:] = 0
        sampled_points[1,:] = sampled_points[1,:] * self.a - self.a / 2
        sampled_points[2,:] = sampled_points[2,:] * self.b - self.b / 2
        sampled_points = self.center + rot_mat @ sampled_points
        
        normal = self.normal
        center = self.center
        normal_line = np.hstack((center, center + normal))
        self.update_sides()
        s1, s2, s3, s4 = self.sides

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_points[0], sampled_points[1], sampled_points[2], c='r', s=1, alpha=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot(s1[0], s1[1], s1[2], c='black')
        ax.plot(s2[0], s2[1], s2[2], c='black')
        ax.plot(s3[0], s3[1], s3[2], c='black')
        ax.plot(s4[0], s4[1], s4[2], c='black')
        ax.plot(normal_line[0], normal_line[1], normal_line[2], c='black')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()
        return fig, ax


# %% Camera class
class Camera():
    def __init__(self, focal_length, sensor_size, image_size, principal_point):
        """
        Parameters:
        - focal_length (2-D list): X and ocal length of the camera.
        """
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.image_size = image_size
        self.principal_point = principal_point

    def get_projection_matrix(self):
        return np.array([[self.focal_length[0], 0, self.principal_point[0]],
                         [0, self.focal_length[1], self.principal_point[1]],
                         [0, 0, 1]])

    def get_extrinsic(self, angles, t_vec):
        """
        Get extrinsic matrix.
        Parameters:
        - angles (2-D list): Rotation angles in x, y, and z directions.
        - t_vec (2-D list): Translation vector.
        Returns:
        - extrinsic (np.array): Extrinsic matrix.
        """
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = self.get_rotation_matrix(angles)
        extrinsic[:3, 3] = t_vec
        return extrinsic

    def project_points(self, points):
        """
        Project 3D points to 2D image plane.
        Parameters:
        - points (np.array): 3D points in world coordinates.
        Returns:
        - points_2d (np.array): 2D points in image coordinates.
        """
        points_2d = np.zeros((points.shape[0], 2))
        for i in range(points.shape[0]):
            points_2d[i] = self.get_projection_matrix() @ points[i]
            points_2d[i] /= points_2d[i][2]
        return points_2d

    def get_rotation_matrix(self, angles):
        """
        Get rotation matrix.
        Parameters:
        - angles (2-D list): Rotation angles in x, y, and z directions.
        Returns:
        - rotation (np.array): Rotation matrix.
        """
        rotation = np.eye(3)
        for i in range(3):
            rotation = rotation @ self.get_rotation_matrix_single(angles[i], i)
        return rotation
    
    def get_rotation_matrix_single(self, angle, axis):
        """
        Get rotation matrix for a single axis.
        Parameters:
        - angle (float): Rotation angle.
        - axis (int): Axis of rotation (0: x, 1: y, 2: z).
        Returns:
        - rotation (np.array): Rotation matrix.
        """
        rotation = np.eye(3)
        if axis == 0:
            rotation[1, 1] = np.cos(angle)
            rotation[1, 2] = -np.sin(angle)
            rotation[2, 1] = np.sin(angle)
            rotation[2, 2] = np.cos(angle)
        elif axis == 1:
            rotation[0, 0] = np.cos(angle)
            rotation[0, 2] = np.sin(angle)
            rotation[2, 0] = -np.sin(angle)
            rotation[2, 2] = np.cos(angle)
        elif axis == 2:
            rotation[0, 0] = np.cos(angle)
            rotation[0, 1] = -np.sin(angle)
            rotation[1, 0] = np.sin(angle)
            rotation[1, 1] = np.cos(angle)
        return rotation
    
# %% Prism class
class Prism():
    def __init__(self, prism_size, prism_vector, prism_center, refractive_index_glass):
        """
        Parameters:
        - prism_size (2-D list): Width and height of the prism.
        - prism_vector (float): Normal of the first surface of the prism. (surface facing the camera)
        - prism_center (2-D list): Center of the first surface of the prism. (surface facing the camera)
        """
        self.prism_size = prism_size
        self.prism_vector = prism_vector
        self.prism_center = prism_center
        self.refractive_index_glass = refractive_index_glass

    def refract_ray(self, ray, normal, n1, n2, intersection):
        """
        Refract a ray
        Parameters:
        - ray (np.array): Ray to refract
        - n1 (float): Refractive index of the first medium
        - n2 (float): Refractive index of the second medium
        Returns:
        - refracted_ray (np.array): Refracted ray
        """
        mat1 = R.align_vectors(np.array([[0, 1, 0]]), normal)
        sinr2 = n1 * np.sqrt(1 - np.dot(ray[1], normal)**2) / n2 # Angle of refraction in rotated frame
        r2 = np.arcsin(sinr2)
        ray_rotated_frame = [np.array([0, 0, 0]), np.array([np.cos(r2), -sinr2, 0])] 
        refracted_ray = [intersection + mat1[0].apply(vec) for vec in ray_rotated_frame]
        return refracted_ray
    
    def get_reflecting_surface(self, prism_size, prism_angle, prism_center):

        return normal, center

    def get_second_surface(self, prism_size, prism_angle, prism_center):

        return normal, center

    def propagate_ray_through_prism(self, ray, prism_size, prism_normal, prism_center):
        """
        Propagate a ray through a prism.
        Parameters:
        - ray (np.array): Ray to propagate.
        - prism_size (2-D list): Width and height of the prism.
        - prism_angle (float): Normal of the first surface of the prism. (surface facing the camera)
        - prism_center (2-D list): Center of the first surface of the prism. (surface facing the camera)
        Returns:
        - refracted_ray (np.array): Refracted ray.
        """
        n1 = 1
        normal_first_surface = prism_normal
        intersection_first_surface = self.get_intersection(ray, self.prism_normal, self.prism_center)
        ray_after_first_surface = self.refract_ray(ray, normal_first_surface, intersection_first_surface, n1, self.refractive_index_glass)
        normal_reflecting_surface, center_reflecting_surface = self.get_reflecting_surface(prism_size, prism_normal, prism_center)
        intersection_reflecting_surface = self.get_intersection(ray_after_first_surface, normal_reflecting_surface, center_reflecting_surface)
        ray_after_reflecting_surface = self.reflect_ray(ray_after_first_surface, intersection_reflecting_surface, normal_reflecting_surface)
        normal_second_surface, center_second_surface = self.get_second_surface(prism_size, self.prism_normal, prism_center)
        intersection_second_surface = self.intersection(ray_after_reflecting_surface, normal_second_surface, center_second_surface)
        ray_after_second_surface = self.refract_ray(ray_after_first_surface, intersection_second_surface, normal_second_surface, self.refractive_index_glass), 1
        return ray_after_second_surface
        

