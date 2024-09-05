"""
Simulates camera, prism and object in a 3-D space 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from config import Config
from scipy.spatial.transform import Rotation as R
#mpl.use('TkAgg')
mpl.use('QtAgg')
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
class Ray():
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
        self.t = 1.
    
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
        self.direction = (point2 - point1) / np.linalg.norm(point2 - point1)
        

    def visualize(self, fig=None, ax=None):
        """
        Visualize the ray.
        """
        t = self.t
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.origin[0], self.origin[1], self.origin[2], marker='o')
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 
                  t * self.direction[0], t * self.direction[1], t * self.direction[2],
                  arrow_length_ratio=0.2)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()
        return fig, ax


# %% Plane class
class Plane():
    def __init__(self, normal=None, center=[0,0,0], alpha=None, beta=None, gamma=None, a=1., b=1.):
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
        
        if not(normal is None) and not(alpha is None or beta is None or gamma is None):
            print('Either provide the normal or all the angles, but not both')
            assert 1==0

        self.a = a
        self.b = b
        
        if normal is None:
            if alpha is None:
                alpha = 0.
            if beta is None:
                beta = 0.
            if gamma is None:
                gamma = 0.

            normal = self.angles_to_normal(alpha=alpha, beta=beta, gamma=gamma)
            
        if not isinstance(normal, np.ndarray):
            normal = np.array(normal)
        if not isinstance(center, np.ndarray):
            center = np.array(center)

        if len(normal.shape) == 1:
            normal = normal.reshape((3, 1))
        
        if len(center.shape) == 1:
            center = center.reshape((3, 1))
        
        self.update_center(center)
        sides = self.angles_to_sides(alpha=alpha, beta=beta, gamma=gamma)
        self.update_normal(normal)
        self.update_sides(sides)
        self.update_angles(alpha, beta, gamma)
        

    def rotate_plane(self, alpha=0., beta=0., gamma=0.):
        normal = self.rotate_normal(normal=self.normal, alpha=alpha,
                                     beta=beta, gamma=gamma)
        sides = self.rotate_sides(side1=self.sides[0], 
                                  side2=self.sides[1],
                                  side3=self.sides[2],
                                  side4=self.sides[3],
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma)
        center = self.rotate_center(self.center, alpha, beta, gamma)
        self.update_center(center)
        alpha, beta, gamma = self.get_angles(normal, sides)
        self.update_normal(normal)
        self.update_angles(alpha, beta, gamma)
        self.update_sides(sides)
        
    def move_plane(self, displacement):
        self.center = self.center + displacement
        for i, side in enumerate(self.sides):
            self.sides[i][:,0] = side[:,0] + displacement[:,0]
            self.sides[i][:,1] = side[:,1] + displacement[:,0]
    
    def rotate_center(self, center, alpha=0., beta=0., gamma=0.):
        Rx = rotx(alpha)
        Ry = roty(beta)
        Rz = rotz(gamma)
        center = Rz @ Ry @ Rx @ center
        return center

    def rotate_normal(self, normal, alpha=0., beta=0., gamma=0.):
        if not isinstance(normal, np.ndarray):
            normal = np.array(normal)
        if len(normal.shape) == 1:
            norma = normal.reshape((3, 1))
        Rx = rotx(alpha)
        Ry = roty(beta)
        Rz = rotz(gamma)
        normal = Rz @ Ry @ Rx @ normal
        return normal

    def angles_to_sides(self, alpha, beta, gamma):
        """
        Get the sides of the plane.
        Returns:
        - sides (2-D list): Four sides of the plane.
        Order of the sides: side parallel to Y-axis and in the positive Z region
                            side parallel to Y-axis and in the negative Z region
                            side parallel to Z-axis and in the positive Y region
                            side parallel to Z-axis and in the negative Y region
        """

        side1 = np.array([[0., self.a/2, self.b/2], [0., -self.a / 2, self.b / 2]]).T 
        side2 = np.array([[0., self.a/2, -self.b/2], [0., -self.a / 2, -self.b / 2]]).T 
        side3 = np.array([[0., self.a/2, self.b/2], [0., self.a/2, -self.b/2]]).T 
        side4 = np.array([[0., -self.a/2, self.b/2], [0., -self.a/2, -self.b/2]]).T 

        if alpha is None:
            alpha = 0
        if beta is None:
            beta = 0
        if gamma is None:
            gamma = 0
        rot_mat = rotz(gamma) @ roty(beta) @ rotx(alpha) # Rotation matrix        
        side1 = self.center + rot_mat @ side1
        side2 = self.center + rot_mat @ side2
        side3 = self.center + rot_mat @ side3
        side4 = self.center + rot_mat @ side4
        return [side1, side2, side3, side4]

    def angles_to_normal(self, alpha=0., beta=0., gamma=0.):
        """
        Reorient unit normal (along X) by the given angles
        Parameters:
        - alpha (float): Angle of the plane with respect to the x-axis.
        - beta (float): Angle of the plane with respect to the y-axis.
        - gamma (float): Angle of the plane with respect to the z-axis.
        """

        normal = np.array([1., 0., 0.])[:, None]
        Rx = rotx(alpha)
        Ry = roty(beta)
        Rz = rotz(gamma)
        normal = Rz @ Ry @ Rx @ normal
        return normal
    
    def get_angles(self, norma=None, sides=None):
        """
        Get the angles of the plane with respect to the x, y, and z axes.
        Given the normal and the sides of the rotated plane.
        Returns:
        - angles (2-D list): Angles of the plane with respect to the x, y, and z axes.
        """

        if sides is None:
            sides = self.sides
        s10 = sides[0][:,0][:, None] - self.center
       
        #print(norma)
        if norma is None:
            norma = self.normal
        if not isinstance(norma, np.ndarray):
            norma = np.array(norma)
        if len(norma.shape) == 1:
            norma = norma.reshape((3, 1))
        # Make sure the normal vector is normalized
        norma = norma / np.linalg.norm(norma)
        gamma = np.arctan2(norma[1], norma[0])[0]
        #print(gamma)
        norma = rotz(-gamma) @ norma
        s10 = rotz(-gamma) @ s10
        beta = np.arctan2(-norma[2], norma[0])[0]
        #print(beta)
        norma = roty(-beta) @ norma
        s10 = roty(-beta) @ s10
        #print(s10)
        assert np.abs(s10[0]) < 1e-5
        alpha = np.arctan2(s10[2], s10[1])[0] - np.arctan2(self.b, self.a)
        return alpha, beta, gamma

    def update_normal(self, normal=[1.,0.,0.]):
        self.normal = normal
    
    def update_angles(self, alpha=0., beta=0., gamma=0.):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update_sides(self, sides):
        self.sides = sides

    def update_center(self, center):
        self.center = center

    def reposition(self, alpha=None, beta=None, gamma=None, center=None):
        """
        Reorient the plane by the given angles
        Set the center of the plane to the given input center array
        Get the equivalent angles of the resulting plane.

        NOTE: alpha, beta, gamma are always defined assuming the default normal vector is [1, 0, 0].
        Parameters:
        - alpha (float): Angle of the plane with respect to the x-axis.
        - beta (float): Angle of the plane with respect to the y-axis.
        - gamma (float): Angle of the plane with respect to the z-axis.
        """
        if alpha is None:
            alpha = 0

        if beta is None:
            beta = 0

        if gamma is None:
            gamma = 0

        if center is None:
            center = self.center
        else:
            if not isinstance(center, np.ndarray):
                center = np.array(center)
            if len(center.shape == 1):
                center = center.reshape((3, 1))
            self.center = center
        
        normal = self.normal
        Rx = rotx(alpha)
        Ry = roty(beta)
        Rz = rotz(gamma)
        self.normal = Rz @ Ry @ Rx @ normal
        self.update_sides(side1=self.sides[0], side2=self.sides[1],
                           side3=self.sides[2], side4=self.sides[3],
                           alpha=alpha, beta=beta, gamma=gamma)
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
        
        t = np.dot(self.center[:,0] - ray_origin, self.normal[:,0]) / np.dot(ray_direction, self.normal[:,0])
        intersection = ray_origin + t * ray_direction
        ray.t = t
        return intersection

    def rotate_sides(self, side1=None, side2=None, side3=None, side4=None, alpha=None, beta=None, gamma=None):
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
            side1 = self.sides[0] 
        else:
            side1 = side1 
        if side2 is None:
            side2 = self.sides[1] 
        else:
            side2 = side2 
        if side3 is None:
            side3 = self.sides[2] 
        else:
            side3 = side3 
        if side4 is None:
            side4 = self.sides[3]
        else:
            side4 = side4 

        if alpha is None:
            alpha = 0
        if beta is None:
            beta = 0
        if gamma is None:
            gamma = 0
        rot_mat = rotz(gamma) @ roty(beta) @ rotx(alpha) # Rotation matrix        
        side1 = rot_mat @ side1
        side2 = rot_mat @ side2
        side3 = rot_mat @ side3
        side4 = rot_mat @ side4
        return [side1, side2, side3, side4]
 
    def visualize(self, fig=None, ax=None, color=[1.,0.,0.]):
        """
        Visualize the plane by plotting the sides and 500 points lying on the plane.
        """
        rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
        sampled_points = np.random.rand(3, 1000)
        sampled_points[0,:] = 0
        sampled_points[1,:] = sampled_points[1,:] * self.a - self.a / 2
        sampled_points[2,:] = sampled_points[2,:] * self.b - self.b / 2
        sampled_points = self.center + rot_mat @ sampled_points
        
        length_of_normal = 0.2 #cm
        normal = self.normal * length_of_normal
        center = self.center
        normal_line = np.hstack((center, center + normal))
        s1, s2, s3, s4 = self.sides

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_points[0], sampled_points[1], sampled_points[2], c=color, s=1, alpha=0.25)
        ax.plot(s1[0], s1[1], s1[2], c='black')
        ax.plot(s2[0], s2[1], s2[2], c='black')
        ax.plot(s3[0], s3[1], s3[2], c='black')
        ax.plot(s4[0], s4[1], s4[2], c='black')
        ax.plot(normal_line[0], normal_line[1], normal_line[2], c='black')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        #plt.show()
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

class OpticalPlane(Plane):
    def __init__(self, alpha=0., beta=0., gamma=0., center=[0.,0.,0.], 
                 normal=None, refractive_idx_1=1., refractive_idx_2=1.,
                  a=1., b=1.):
        super().__init__(normal=normal, center=center, alpha=alpha, beta=beta, gamma=gamma, a=a, b=b)

        self.refractive_idx_1 = refractive_idx_1
        self.refractive_idx_2 = refractive_idx_2

    def reflect_ray(self, ray):
        intersection = self.get_intersection(ray)
        normal = self.normal[:,0]
        cosi = np.dot(ray.direction, normal)
        if cosi < 0:
            normal = -normal
            cosi = -cosi
        displacement = 2 * (ray.direction - cosi * normal)
        reflected_ray = Ray()
        reflected_ray.origin = intersection
        reflected_ray.direction = displacement - ray.direction
        return reflected_ray

    def refract_ray(self, ray):
        """
        Refract a ray
        Parameters:
        - ray (np.array): Ray to refract
        - n1 (float): Refractive index of the first medium
        - n2 (float): Refractive index of the second medium
        Returns:
        - refracted_ray (np.array): Refracted ray
        """
        #print(f'Input ray direction: {ray.direction}')
        intersection = self.get_intersection(ray)
        normal = self.normal[:,0]
        refractive_idx_1 = self.refractive_idx_1
        refractive_idx_2 = self.refractive_idx_2
        #mat1 = R.align_vectors(np.array([[0, 1, 0]]), normal[None, :])
        
        #incoming_ray_vertical = mat1[0].apply(ray.direction)
        #r2 = np.arctan2(incoming_ray_vertical[1], incoming_ray_vertical[0])
        cosi = np.dot(ray.direction, normal) #angle of incidence
        cosi = np.clip(cosi, -1., 1.) # floating point errors can lead to cosi being slightly outside [-1, 1]
        if cosi < 0: # ray within the prism
            normal = -normal
            cosi = -cosi
        else:
            temp = refractive_idx_1
            refractive_idx_1 = refractive_idx_2
            refractive_idx_2 = temp
        sini = np.sqrt(1 - cosi**2)
        sinr = refractive_idx_1 * sini / refractive_idx_2
        cosr = np.sqrt(1 - sinr**2)
        ray_displacement = refractive_idx_1 / refractive_idx_2 * (cosr - cosi) * normal
        #print(f'cosr {cosr}, cosi {cosi}, sini {sini}, sinr {sinr}, ')
        #print(f'r {180/np.pi * np.arcsin(sinr)}, i {180/np.pi * np.arcsin(sini)}')
        #sinr2 = refractive_idx_1 * np.sqrt(1 - np.dot(ray.direction, normal)**2) / refractive_idx_2 # Angle of refraction in rotated frame  
        #sinr2 = sinr2[0]
        #r2 = np.arcsin(sinr2)
        
        #ray_rotated_frame = [np.array([0, 0, 0]), np.array([sinr2, -np.cos(r2), 0])] 

        #refracted_ray_ = [intersection + mat1[0].apply(vec) for vec in ray_rotated_frame]
        refracted_ray = Ray()
        refracted_ray.origin = intersection
        refracted_ray.direction = ray.direction + ray_displacement
        #print(f'Input ray direction: {ray.direction}')
        #print(f'Refracted ray direction: {refracted_ray.direction}')
        return refracted_ray


# %% Prism class
class Prism():

    def __init__(self, prism_size=[1.,1.,1.], prism_angles=[0.,0.,0.], prism_center=[0.,0.,0.], refractive_index_glass=1.5, refractive_index_air=1.):
        """
        Parameters:
        - prism_size (list): Length (X), width (Z) and height (Y) of the prism.
        - prism_angles (list): A list of angles alpha (X-axis), beta (Y-axis), gamma (Z-axis)
        - prism_center (list): Center of the first surface of the prism. (surface facing the camera)
        """
        
        self.prism_size = prism_size
        self.prism_angles = prism_angles
        self.prism_center = prism_center
        self.refractive_index_glass = refractive_index_glass
        n_air = refractive_index_air
        n_glass = refractive_index_glass
        prism_alpha, prism_beta, prism_gamma = prism_angles
        plane1 = OpticalPlane(refractive_idx_1=n_air, refractive_idx_2=n_glass, a=1., b=1.) # Plane facing the camera
        plane2_center = np.array([plane1.center[0,0] - plane1.a/2,
                                plane1.center[1,0],
                                plane1.center[2,0]])[:, None]
        plane2_b = plane1.b * np.sqrt(2)
        plane2 = OpticalPlane(refractive_idx_1=n_air, refractive_idx_2=n_glass, 
                              beta=np.pi/4 + np.pi, a = plane1.a, b=plane2_b, center=plane2_center)
        
        plane3_center = np.array([plane1.center[0,0] - plane1.a/2, 
                                plane1.center[1,0],
                                plane1.center[2,0] - plane1.b/2])
        plane3 = OpticalPlane(refractive_idx_1=n_air, refractive_idx_2=n_glass, 
                              beta=np.pi/2, a=plane1.a, b=plane1.b, center=plane3_center)

        plane1.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane2.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane3.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane1.move_plane(prism_center)
        plane2.move_plane(prism_center)
        plane3.move_plane(prism_center)
        self.plane1 = plane1
        self.plane2 = plane2
        self.plane3 = plane3


    def trace_ray(self, origin_point=[0.,0.6,0.25], target_point=[0.,0.4,0.]):
        self.ray1 = Ray()
        self.ray1.build_ray(origin_point, target_point)            
        self.ray2 = self.plane1.refract_ray(self.ray1)
        self.ray3 = self.plane2.reflect_ray(self.ray2)
        self.ray4 = self.plane3.refract_ray(self.ray3)
        #fig = plt.figure(figsize=(10,10))
        #ax = fig.add_subplot(111, projection='3d')
        #fig, ax = self.plane1.visualize(fig, ax)
        #fig, ax = self.plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        #fig, ax = self.plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        #fig, ax = self.ray1.visualize(fig, ax)
        #fig, ax = self.ray2.visualize(fig, ax)
        #fig, ax = self.ray3.visualize(fig, ax)
        #self.ray4.visualize(fig, ax)
    
    def visualize_prism(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        fig, ax = self.plane1.visualize(fig, ax)
        fig, ax = self.plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        fig, ax = self.plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        return fig, ax
    
    def visualize_prism_and_ray(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        fig, ax = self.plane1.visualize(fig, ax)
        fig, ax = self.plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        fig, ax = self.plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = self.ray1.visualize(fig, ax)
        fig, ax = self.ray2.visualize(fig, ax)
        fig, ax = self.ray3.visualize(fig, ax)
        fig, ax = self.ray4.visualize(fig, ax)
        return fig, ax
    
    
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
        

optical = True
if __name__=="__main__":
    if optical:
        n_glass = 1.55
        n_air = 1.
        prism_alpha = 0.
        prism_beta = -np.pi / 2
        prism_gamma = np.pi / 2 
        prism_center = np.array([0.,0.,0.])[:, None]
        
        prism = Prism(prism_size=[1.,1.,1.], prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                      prism_center=prism_center, refractive_index_glass=n_glass, 
                      refractive_index_air=n_air)
        
        
        origin_point=[0.,0.6,0.25] 
        target_point=[0.,0.4,0.]
        prism.trace_ray(origin_point, target_point) 
        fig, ax = prism.visualize_prism_and_ray()
        ax.set_aspect('equal', adjustable='datalim')        
        plt.show()
 
    else: 
        print('Not optical')
        prism_alpha = 0 * np.pi 
        prism_beta = 0.
        prism_gamma = np.pi / 2
        plane1 = Plane()
        plane2_center = np.array([plane1.center[0,0] - plane1.a/2,
                                plane1.center[1,0],
                                plane1.center[2,0]])[:, None]
        plane2_b = plane1.b * np.sqrt(2)
        plane2 = Plane(beta=np.pi/4, a = plane1.a, b=plane2_b, center=plane2_center)
        
        plane3_center = np.array([plane1.center[0,0] - plane1.a/2, 
                                plane1.center[1,0],
                                plane1.center[2,0] - plane1.b/2])
        plane3 = Plane(beta=np.pi/2, a=plane1.a, b=plane1.b, center=plane3_center)

        plane1.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane2.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane3.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        fig, ax = plane1.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        plane3.visualize(fig, ax)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
