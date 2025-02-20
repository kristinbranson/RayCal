"""
Simulates camera, prism and object in a 3-D space 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from config import Config
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
mpl.use('TkAgg') # Use this if working on the PC
#mpl.use('QtAgg') # Use this if working remotely with NoMachine
plt.ion()

pi = torch.tensor(np.pi)

def rotx(angle):
    """
    Rotation matrix around x-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (np.array): Rotation matrix.
    """
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    rotation = torch.tensor([[1, 0, 0],
                          [0, torch.cos(angle), -torch.sin(angle)],
                          [0, torch.sin(angle), torch.cos(angle)]])
    return rotation

def roty(angle):
    """
    Rotation matrix around y-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (np.array): Rotation matrix.
    """
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    rotation = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)],
                          [0, 1, 0],
                          [-torch.sin(angle), 0, torch.cos(angle)]])
    return rotation

def rotz(angle):
    """
    Rotation matrix around z-axis.
    Parameters:
    - angle (float): Angle of rotation.
    Returns:
    - rotation (torch.tensor): Rotation matrix.
    """
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                          [torch.sin(angle), torch.cos(angle), 0],
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
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32)
            if len(origin.shape) == 1:
                origin = origin.reshape((3, 1))
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32)
            if len(direction.shape) == 1:
                direction = direction.reshape((3, 1))
        
        direction = direction / torch.linalg.vector_norm(direction, dim=0).to(torch.float32)
        self.origin = origin
        self.direction = direction
        self.t = torch.ones((self.direction.shape[1], 1))
    
    def build_ray(self, point1, point2):
        """
        Build a ray from two points.
        Parameters:
        - point1 (2-D list): First point (this will become the origin of the ray).
        - point2 (2-D list): Second point.
        """
        if not isinstance(point1, torch.Tensor):
            point1 = torch.tensor(point1, dtype=torch.float32)
            if len(point1.shape) == 1:
                point1 = point1.reshape((3, 1))
        if not isinstance(point2, torch.Tensor):
            point2 = torch.tensor(point2, dtype=torch.float32)
            if len(point2.shape) == 1:
                point2 = point2.reshape((3, 1))

        self.origin = point1
        self.direction = (point2 - point1) / torch.linalg.vector_norm(point2 - point1, dim=0)
        self.t = torch.ones((self.direction.shape[1], 1))
        
    def distance_to_point(self, point):
        """
        Get the distance of the ray to a point.
        Parameters:
        - point (2-D list): Point.
        Returns:
        - distance (2-D list): Distance of the ray to the point.
        """
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, dtype=torch.float32)
            if len(point.shape) == 1:
                point = point.reshape((3, 1))
        distance = torch.linalg.cross(self.direction, (point - self.origin), dim=0)
        distance = torch.linalg.norm(distance, dim=0)
        return distance

    def visualize(self, fig=None, ax=None):
        """
        Visualize the ray.
        """
        t = self.t.detach().numpy()
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.origin[0].detach().numpy(), 
                   self.origin[1].detach().numpy(), 
                   self.origin[2].detach().numpy(), 
                   s=1, 
                   marker='o', 
                   color='black')
        #NOTE: t has a shape of (N,1) where N is the number of rays
        for ray_id in range(self.t.shape[0]):
            ax.plot([self.origin[0, ray_id].detach().numpy(),
             (self.origin[0, ray_id] + t[ray_id, 0] * self.direction[0, ray_id]).detach().numpy()],
             [self.origin[1, ray_id].detach().numpy(), (self.origin[1, ray_id] + t[ray_id, 0] * self.direction[1, ray_id]).detach().numpy()],
             [self.origin[2, ray_id].detach().numpy(), (self.origin[2, ray_id] + t[ray_id, 0] * self.direction[2, ray_id]).detach().numpy()],
             color='black', linewidth=0.2)

        ax.quiver(self.origin[0].detach().numpy(), 
                  self.origin[1].detach().numpy(), 
                  self.origin[2].detach().numpy(), 
                  t * self.direction[0].detach().numpy(), 
                  t * self.direction[1].detach().numpy(), 
                  t * self.direction[2].detach().numpy(),
                  length=0.2, linewidth=0.1, color='black', normalize=True)
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

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)

        self.a = nn.Parameter(a, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=torch.float32)
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=torch.float32)

        if normal is None:
            if alpha is None:
                alpha = 0.
            if beta is None:
                beta = 0.
            if gamma is None:
                gamma = 0.
            normal = self.angles_to_normal(alpha=alpha, beta=beta, gamma=gamma)
            
        if not isinstance(normal, torch.Tensor):
            normal = torch.tensor(normal, dtype=torch.float32)
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)

        if len(normal.shape) == 1:
            normal = normal.reshape((3, 1))
        
        if len(center.shape) == 1:
            center = center.reshape((3, 1))                        
    
        self.alpha = nn.Parameter(alpha, requires_grad=True)
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma = nn.Parameter(gamma, requires_grad=True)
        self.center = center.to(torch.float32)
        self.center = nn.Parameter(center, requires_grad=True)
        #self.normal = normal
        #self.update_sides(sides)
        self.update_angles(alpha, beta, gamma)
    
        #self.horizontal_direction = self.get_horizontal_direction()
        #self.vertical_direction = self.get_vertical_direction()
        
    @property
    def normal(self):
        return self.angles_to_normal(self.alpha, self.beta, self.gamma)

    
    @property
    def sides(self):
        return self.angles_to_sides(self.alpha, self.beta, self.gamma)

    @property
    def horizontal_direction(self):
        return self.get_horizontal_direction()
    
    @property
    def vertical_direction(self):
        return self.get_vertical_direction()
    
    def get_horizontal_direction(self, original_horizontal_direction=[0., 0., -1.], rot_mat=None):
        #NOTE: You need the original vertical direction in case the plane has been rotated
        if not isinstance(original_horizontal_direction, torch.Tensor):
            original_horizontal_direction = torch.tensor(original_horizontal_direction, dtype=torch.float32)
        if len(original_horizontal_direction.shape) == 1:
            original_horizontal_direction = original_horizontal_direction.unsqueeze(-1)
        #horizontal_direction = torch.tensor([0., 0., -1.], dtype=torch.float32).unsqueeze(-1)
        if rot_mat is None:
            #rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
            rot_mat = self.get_rot_mat(self.alpha, self.beta, self.gamma)
        horizontal_direction = rot_mat @ original_horizontal_direction
        return horizontal_direction
    
    

    def get_vertical_direction(self, original_vertical_direction=[0., 1., 0.], rot_mat=None):
        #NOTE: You need the original vertical direction in case the plane has been rotated
        if not isinstance(original_vertical_direction, torch.Tensor):
            original_vertical_direction = torch.tensor(original_vertical_direction, dtype=torch.float32)
        if len(original_vertical_direction.shape) == 1:
            original_vertical_direction = original_vertical_direction.unsqueeze(-1)
        #vertical_direction = torch.tensor([0., 1., 0.], dtype=torch.float32).unsqueeze(-1)
        if rot_mat is None:
            #rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
            rot_mat = self.get_rot_mat(self.alpha, self.beta, self.gamma)
        vertical_direction = rot_mat @ original_vertical_direction
        return vertical_direction

    def rotate_plane(self, alpha=0., beta=0., gamma=0., rot_mat=None):
        rotated_normal = self.rotate_normal(normal=self.normal, alpha=alpha,
                                     beta=beta, gamma=gamma, rot_mat=rot_mat)
        rotated_sides = self.rotate_sides(side1=self.sides[0],
                                  side2=self.sides[1],
                                  side3=self.sides[2],
                                  side4=self.sides[3],
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  rot_mat=rot_mat)
        self.center = self.rotate_center(self.center, alpha, beta, gamma, rot_mat=rot_mat)        
        alpha, beta, gamma = self.get_angles(rotated_normal, rotated_sides)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        #self.horizontal_direction = self.get_horizontal_direction(
        #    original_horizontal_direction=self.horizontal_direction,
        #    rot_mat=rot_mat) # Note that angles have been updated first
        #self.vertical_direction = self.get_vertical_direction(
        #    original_vertical_direction=self.vertical_direction, 
        #    rot_mat=rot_mat) # Note that angles have been updated first
        #self.update_sides(sides)
        
    def move_plane(self, displacement):
        self.center = self.center + displacement
        #for i, side in enumerate(self.sides):
        #    # Side is defined by two end points. Displace each end point by 'displacement'
        #    self.sides[i][:,0] = side[:,0] + displacement[:,0]
        #    self.sides[i][:,1] = side[:,1] + displacement[:,0]
    
    def rotate_center(self, center, alpha=0., beta=0., gamma=0., rot_mat=None):
        if rot_mat is None:    
            rot_mat = self.get_rot_mat(alpha, beta, gamma) # Rotation matrix
        center = rot_mat @ center
        return center

    def rotate_normal(self, normal, alpha=0., beta=0., gamma=0., rot_mat=None):
        if not isinstance(normal, torch.Tensor):
            normal = torch.tensor(normal)
        if len(normal.shape) == 1:
            normal = normal.reshape((3, 1))
        if rot_mat is None:
            rot_mat = self.get_rot_mat(alpha, beta, gamma) # Rotation matrix
        
        normal = rot_mat @ normal
        return normal

    def angles_to_sides(self, alpha, beta, gamma):
        """
        Get the sides of the plane.
        Uses the alpha, beta, gamma and center attribute of the plane to estimate the sides
        Returns:
        - sides (2-D list): Four sides of the plane.
        Order of the sides: side parallel to Y-axis and in the positive Z region
                            side parallel to Y-axis and in the negative Z region
                            side parallel to Z-axis and in the positive Y region
                            side parallel to Z-axis and in the negative Y region
        """

        side1 = torch.tensor([[0., self.a/2, self.b/2], [0., -self.a / 2, self.b / 2]]).T 
        side2 = torch.tensor([[0., self.a/2, -self.b/2], [0., -self.a / 2, -self.b / 2]]).T 
        side3 = torch.tensor([[0., self.a/2, self.b/2], [0., self.a/2, -self.b/2]]).T 
        side4 = torch.tensor([[0., -self.a/2, self.b/2], [0., -self.a/2, -self.b/2]]).T 

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

        normal = torch.tensor([1., 0., 0.]).unsqueeze(-1)
        Rx = rotx(alpha)
        Ry = roty(beta)
        Rz = rotz(gamma)
        normal = Rz @ Ry @ Rx @ normal
        return normal
    
    def get_angles(self, normal=None, sides=None):
        """
        Get the angles of the plane with respect to the x, y, and z axes.
        Given the normal and the sides of the rotated plane.
        Returns:
        - angles (2-D list): Angles of the plane with respect to the x, y, and z axes.
        """

        if sides is None:
            sides = self.sides
        s10 = sides[0][:,0].unsqueeze(-1) - self.center       

        if normal is None:
            normal = self.normal
        if not isinstance(normal, torch.Tensor):
            normal = torch.tensor(normal)
        if len(normal.shape) == 1:
            normal = normal.reshape((3, 1))
        
        # Make sure the normal vector is normalized
        normal = normal / torch.linalg.vector_norm(normal)
        gamma = torch.arctan2(normal[1], normal[0])[0]

        normal = rotz(-gamma) @ normal
        s10 = rotz(-gamma) @ s10
        beta = torch.arctan2(-normal[2], normal[0])[0]

        normal = roty(-beta) @ normal
        s10 = roty(-beta) @ s10

        assert torch.abs(s10[0]) < 1e-5
        alpha = torch.arctan2(s10[2], s10[1])[0] - torch.arctan2(self.b, self.a)
        return alpha, beta, gamma

    
    def update_angles(self, alpha=0., beta=0., gamma=0.):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update_sides(self, sides):
        self.sides = sides

    def update_center(self, center):
        self.center = center.to(torch.float32)

    def get_rot_mat(self, alpha=0., beta=0., gamma=0.):
        return rotz(gamma) @ roty(beta) @ rotx(alpha) # Rotation matrix

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
        if alpha is None:
            alpha = 0

        if beta is None:
            beta = 0

        if gamma is None:
            gamma = 0

        if center is None:
            center = self.center
        else:
            if not isinstance(center, torch.Tensor):
                center = torch.tensor(center)
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
        """
        pass        
        
        

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
        
        t = torch.mm((self.center - ray_origin).T, self.normal) / torch.mm(ray_direction.T, self.normal)
        intersection = ray_origin + t.T * ray_direction
        ray.t = t
        return intersection

    def rotate_sides(self, side1=None, side2=None, side3=None, side4=None, alpha=None, beta=None, gamma=None, rot_mat=None):
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
        if rot_mat is None:
            self.get_rot_mat(alpha, beta, gamma)            
            rot_mat = self.get_rot_mat(alpha, beta, gamma) # Rotation matrix

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
        sampled_points = torch.rand(3, 500)
        sampled_points[0,:] = 0
        sampled_points[1,:] = sampled_points[1,:] * self.a - self.a / 2
        sampled_points[2,:] = sampled_points[2,:] * self.b - self.b / 2
        sampled_points = self.center + rot_mat @ sampled_points
        
        length_of_normal = 0.2 #cm
        normal = self.normal * length_of_normal
        center = self.center
        normal_line = torch.hstack((center, center + normal))
        s1, s2, s3, s4 = self.sides

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_points[0].detach().numpy(),
                    sampled_points[1].detach().numpy(), 
                    sampled_points[2].detach().numpy(),
                    c=color,
                    s=1,
                    alpha=0.25)
        ax.plot(s1[0].detach().numpy(), s1[1].detach().numpy(), s1[2].detach().numpy(), c='black')
        ax.plot(s2[0].detach().numpy(), s2[1].detach().numpy(), s2[2].detach().numpy(), c='black')
        ax.plot(s3[0].detach().numpy(), s3[1].detach().numpy(), s3[2].detach().numpy(), c='black')
        ax.plot(s4[0].detach().numpy(), s4[1].detach().numpy(), s4[2].detach().numpy(), c='black')
        ax.plot(normal_line[0].detach().numpy(),
                normal_line[1].detach().numpy(), 
                normal_line[2].detach().numpy(), 
                c='black')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        #plt.show()
        return fig, ax

#%% Camera  class
class Camera(Plane):
    # NOTE: The plane defines the camera sensor, not the aperture plane
    def __init__(self, alpha=0, beta=-pi/2, gamma=0., aperture=[0.,0.,0.], 
                 normal=None, height=4.14, width=6.624, focal_length=19.65, pixel_size=3.45e-3, 
                 principal_point_pixel=None):        
                 
        super().__init__(normal=normal, center=[0.,0.,0], alpha=alpha, beta=beta, gamma=gamma, a=height, b=width)
        
        if principal_point_pixel is None:
            principal_point_pixel = torch.tensor([width/pixel_size/2, height/pixel_size/2.], dtype=torch.float32)[:, None]

        if not isinstance(aperture, torch.Tensor):
            aperture = torch.tensor(aperture)
            if len(aperture.shape) == 1:
                aperture = aperture.reshape((3, 1))

        if not isinstance(principal_point_pixel, torch.Tensor):
            principal_point_pixel = torch.tensor(principal_point_pixel, dtype=torch.float32)
            if len(principal_point_pixel.shape) == 1:
                principal_point_pixel = principal_point_pixel.reshape((2, 1))

        self.aperture = aperture
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.principal_point_pixel = principal_point_pixel
        principal_point = self.get_principal_point_from_aperture()
        self.move_plane(displacement=principal_point - aperture)

        
        #NOTE: For the camera, horizontal plane direction is the self.vertical_direction 
        #self.horizontal_direction = self.get_horizontal_direction()
        #self.vertical_direction = self.get_vertical_direction()

    def get_principal_point_from_aperture(self):
        return self.aperture - self.focal_length * self.normal
        
    def initialize_ray(self, pixel):
        """
        Converts pixel coordinates to world coordinates and initializes a ray.
        """
        if not isinstance(pixel, torch.Tensor):
            pixel = torch.tensor(pixel, dtype=torch.float32)
            if len(pixel.shape) == 1:
                pixel = pixel.reshape((2, 1))
        pixels = self.pixels_to_world(pixel)
        ray = Ray()
        aperture = self.aperture.repeat(1, pixels.shape[1]).clone()
        ray.build_ray(pixels, aperture)
        return ray
    
    def pixels_to_world(self, digital_pixels):
        """
        Convert pixel coordinates to world coordinates.
        pixels: (N,2) array of pixel coordinates
        """
        digital_pixels = digital_pixels - self.principal_point_pixel #NOTE: Center is the principal point
        digital_pixels = digital_pixels * self.pixel_size
        #digital_pixels[0,:] = -digital_pixels[0,:]
        #digital_pixels[1,:] = -digital_pixels[1,:] # Flip the y-axis
        # NOTE: For the camera, horizontal plane direction is the self.horizontal_direction
        pixels = digital_pixels[0,:] * (-1) * self.horizontal_direction + digital_pixels[1,:] * (-1) * self.vertical_direction
        pixels = pixels + self.center
        return pixels
    
    def update_camera_pose(self, R, T):
        """
        Update the camera pose, given the camera extrinsics as Rotation Matrix (R) and Translation vector (t)
        """
        self.move_plane(displacement=-T)
        self.rotate_plane(rot_mat=R)        
        self.aperture = self.center + self.focal_length * self.normal



def visualize_camera_configuration(camera=None, prism=None, pixels=None, ax=None, fig=None):
    """
    Visualize the camera configuration.
    """
    if camera is None:
        camera = Camera()
    
    prism_distance = 130.

    if prism is None:
        prism_center = camera.aperture + camera.normal * prism_distance
        prism = Prism(prism_size=[30.,30.,30.], prism_center=prism_center)
            
    if pixels is None:
        pixels = torch.tensor([1,1]).unsqueeze(-1)
    
    ray = camera.initialize_ray(pixels)
    prism.trace_ray(ray.origin, ray.origin + ray.direction)
    fig, ax = camera.visualize(fig=fig, ax=ax)
    fig, ax = prism.visualize_prism_and_ray(fig=fig, ax=ax)
    ax.scatter(camera.aperture[0].detach().numpy(), 
               camera.aperture[1].detach().numpy(), 
               camera.aperture[2].detach().numpy(), 
               c='black', s=10)
    ax.set_aspect('equal', adjustable='datalim')        
    return fig, ax, prism, camera

# %% Clas OpticalPlane

class OpticalPlane(Plane):
    def __init__(self, alpha=0., beta=0., gamma=0., center=[0.,0.,0.], 
                 normal=None, refractive_idx_1=1., refractive_idx_2=1.,
                  a=1., b=1.):
        super().__init__(normal=normal, center=center, alpha=alpha, beta=beta, gamma=gamma, a=a, b=b)

        if not isinstance(refractive_idx_1, torch.Tensor):
            refractive_idx_1 = torch.tensor(refractive_idx_1)
        if not isinstance(refractive_idx_2, torch.Tensor):
            refractive_idx_2 = torch.tensor(refractive_idx_2)

        self.refractive_idx_1 = nn.Parameter(refractive_idx_1, requires_grad=True)
        self.refractive_idx_2 = nn.Parameter(refractive_idx_2, requires_grad=True)

    def reflect_ray(self, ray):
        if self.center.dtype != ray.origin.dtype:
            self.center = self.center.to(ray.origin.dtype)
        intersection = self.get_intersection(ray)
        normal = self.normal
        cosi = torch.mm(ray.direction.T, normal).T

        normal_multiplier = torch.ones(cosi.shape)
        normal_multiplier[cosi < 0] = -1
        
#        if cosi < 0:
#            normal = -normal
        cosi = torch.abs(cosi)
        displacement = 2 * normal * (ray.direction - cosi * normal_multiplier) 
        #displacement = displacement.T
        reflected_ray = Ray()
        reflected_ray.origin = intersection
        reflected_ray.direction = displacement - ray.direction
        reflected_ray.direction = reflected_ray.direction / torch.linalg.vector_norm(reflected_ray.direction, dim=0)
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
        if self.center.dtype != ray.origin.dtype:
            self.center = self.center.to(ray.origin.dtype)
        intersection = self.get_intersection(ray)
        normal = self.normal
        refractive_idx_1 = self.refractive_idx_1
        refractive_idx_2 = self.refractive_idx_2
        #mat1 = R.align_vectors(np.array([[0, 1, 0]]), normal[None, :])
        
        #incoming_ray_vertical = mat1[0].apply(ray.direction)
        #r2 = np.arctan2(incoming_ray_vertical[1], incoming_ray_vertical[0])
        cosi = torch.mm(ray.direction.T, normal).T #angle of incidence
        cosi = torch.clip(cosi, -1., 1.) # floating point errors can lead to cosi being slightly outside [-1, 1]
        #normal_inverter = np.zeros(cosi.shape)
        #normal_inverter[cosi < 0] = -1
        normal_multiplier = torch.ones(cosi.shape)
        normal_multiplier[cosi < 0] = -1
        refractive_idx_1 = refractive_idx_1.clone() * torch.ones(cosi.shape)
        refractive_idx_2 = refractive_idx_2.clone() * torch.ones(cosi.shape)
        temp = refractive_idx_1.clone()
        refractive_idx_1[cosi > 0] = refractive_idx_2[cosi > 0]
        refractive_idx_2[cosi > 0] = temp[cosi > 0]

        
        cosi = torch.abs(cosi)        
        sini = torch.sqrt(1 - cosi**2)
        sinr = refractive_idx_1 * sini / refractive_idx_2

        cosr = torch.sqrt(1 - sinr**2)
        ray_displacement = normal * (refractive_idx_1 / refractive_idx_2 * (cosr - cosi) * normal_multiplier) 
        
        refracted_ray_direction = ray.direction + ray_displacement
        refracted_ray = Ray(origin=intersection, direction=refracted_ray_direction / torch.linalg.vector_norm(refracted_ray_direction, dim=0))

        return refracted_ray


# %% Prism class
class Prism():

    def __init__(self, prism_size=[1.,1.,1.], prism_angles=[0.,pi/2,-pi/2], prism_center=[0.,0.,0.], refractive_index_glass=1.5, refractive_index_air=1.):
        """
        Parameters:`
        - prism_size (list): Width (X), Height (Y) and Depth (Z) of the prism.
        - prism_angles (list): A list of angles alpha (X-axis), beta (Y-axis), gamma (Z-axis)
        - prism_center (list): Center of the first surface of the prism. (surface facing the camera)
        """
        
        if not isinstance(prism_center, torch.Tensor):
            prism_center = torch.tensor(prism_center)
            if len(prism_center.shape) == 1:
                prism_center = prism_center.reshape((3, 1))                

        self.prism_size = prism_size
        self.prism_angles = prism_angles
        self.prism_center = prism_center
        self.refractive_index_glass = refractive_index_glass
        self.refractive_index_air = refractive_index_air
        #n_air = refractive_index_air
        #n_glass = refractive_index_glass
        #prism_alpha, prism_beta, prism_gamma = prism_angles
        #self.updatePrism()
        
    @property
    def plane1(self):                
        prism_alpha, prism_beta, prism_gamma = self.prism_angles
        plane1 = OpticalPlane(refractive_idx_1=self.refractive_index_air,
                                    refractive_idx_2=self.refractive_index_glass,
                               a=self.prism_size[0], b=self.prism_size[1]) # Plane facing the camera
        plane1.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane1.move_plane(self.prism_center)                    
        return plane1
    
    @property
    def plane2(self):
        ref_plane1 = Prism().plane1
        plane2_center = torch.tensor([ref_plane1.center[0,0] - ref_plane1.a/2,
                                ref_plane1.center[1,0],
                                ref_plane1.center[2,0]]).unsqueeze(-1)
        plane2_b = ref_plane1.b * torch.sqrt(torch.tensor(2))
        prism_alpha, prism_beta, prism_gamma = self.prism_angles
        plane2 = OpticalPlane(refractive_idx_1=self.refractive_index_air, refractive_idx_2=self.refractive_index_glass, 
                              beta=pi/4 + pi, a = ref_plane1.a, b=plane2_b, center=plane2_center)
        plane2.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane2.move_plane(self.prism_center)
        return plane2
    
    @property
    def plane3(self):
        ref_plane1 = Prism().plane1
        self.plane3_center = plane3_center = torch.tensor([ref_plane1.center[0,0] - ref_plane1.a/2, 
                                ref_plane1.center[1,0],
                                ref_plane1.center[2,0] - ref_plane1.b/2])
        plane3 = OpticalPlane(refractive_idx_1=self.refractive_index_air, 
                                   refractive_idx_2=self.refractive_index_glass, 
                              beta=pi/2, a=self.prism_size[0], b=self.prism_size[2], center=plane3_center)
        prism_alpha, prism_beta, prism_gamma = self.prism_angles
        plane3.rotate_plane(alpha=prism_alpha,
                            beta=prism_beta,
                            gamma=prism_gamma)
        plane3.move_plane(self.prism_center)
        return plane3


    def updatePrism(self):
        #self.plane1 = OpticalPlane(refractive_idx_1=self.refractive_index_air,
        #                            refractive_idx_2=self.refractive_index_glass,
        #                       a=self.prism_size[0], b=self.prism_size[1]) # Plane facing the camera
        #plane2_center = torch.tensor([self.plane1.center[0,0] - self.plane1.a/2,
        #                        self.plane1.center[1,0],
        #                        self.plane1.center[2,0]]).unsqueeze(-1)
        #plane2_b = self.plane1.b * torch.sqrt(torch.tensor(2))
        #self.plane2 = OpticalPlane(refractive_idx_1=self.refractive_index_air, refractive_idx_2=self.refractive_index_glass, 
        #                      beta=pi/4 + pi, a = self.plane1.a, b=plane2_b, center=plane2_center)
        
        #plane3_center = torch.tensor([self.plane1.center[0,0] - self.plane1.a/2, 
        #                        self.plane1.center[1,0],
        #                        self.plane1.center[2,0] - self.plane1.b/2])
        #self.plane3 = OpticalPlane(refractive_idx_1=self.refractive_index_air, 
        #                           refractive_idx_2=self.refractive_index_glass, 
        #                      beta=pi/2, a=self.prism_size[0], b=self.prism_size[2], center=plane3_center)
        
        #prism_alpha, prism_beta, prism_gamma = self.prism_angles

        #self.plane1.rotate_plane(alpha=prism_alpha,
        #                    beta=prism_beta,
        #                    gamma=prism_gamma)
        #self.plane2.rotate_plane(alpha=prism_alpha,
        #                    beta=prism_beta,
        #                    gamma=prism_gamma)
        #self.plane3.rotate_plane(alpha=prism_alpha,
        #                    beta=prism_beta,
        #                    gamma=prism_gamma)
        #self.plane1.move_plane(self.prism_center)
        #self.plane2.move_plane(self.prism_center)
        #self.plane3.move_plane(self.prism_center)
        pass


    def trace_ray(self, origin_point=[0.,0.6,0.25], target_point=[0.,0.4,0.]):
        "target_point is, for example, the camera's aperture"
        if not isinstance(origin_point, torch.Tensor):
            origin_point = torch.tensor(origin_point)
            if len(origin_point.shape) == 1:
                origin_point = origin_point.reshape((3, 1))

        if not isinstance(target_point, torch.Tensor):
            target_point = torch.tensor(target_point)
            if len(target_point.shape) == 1:
                target_point = target_point.reshape((3, 1))        
        
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
        fig, ax = self.plane1.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = self.plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        fig, ax = self.plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = self.ray1.visualize(fig, ax)
        fig, ax = self.ray2.visualize(fig, ax)
        fig, ax = self.ray3.visualize(fig, ax)
        fig, ax = self.ray4.visualize(fig, ax)
        return fig, ax
    
def closest_point(ray1, ray2):
    """
    Returns the closest point between two rays, and the closest distance between the rays
    """
    n = torch.linalg.cross(ray1.direction, ray2.direction, dim=0)
    n2 = torch.linalg.cross(ray2.direction, n, dim=0)
    n1 = torch.linalg.cross(ray1.direction, n, dim=0)

    c1 = ray1.origin + ((torch.linalg.vecdot(ray2.origin - ray1.origin, n2, dim=0)) / torch.linalg.vecdot(ray1.direction, n2, dim=0).unsqueeze(0)) * ray1.direction
    c2 = ray2.origin + ((torch.linalg.vecdot(ray1.origin - ray2.origin, n1, dim=0)) / torch.linalg.vecdot(ray2.direction, n1, dim=0).unsqueeze(0)) * ray2.direction
    return (c1 + c2) / 2, torch.linalg.norm(c1 - c2, dim=0)


# %% Arena class
class Arena(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance):
        super(Arena, self).__init__()
        # Camera initialization        
        self.camera1 = Camera(principal_point_pixel=principal_point_pixel_cam_0, focal_length=focal_length_cam_0)
        self.camera2 = Camera(principal_point_pixel=principal_point_pixel_cam_1, focal_length=focal_length_cam_1)
        self.camera2.update_camera_pose(R, T)
        # Prism initialization
        prism_center = self.camera1.aperture + self.camera1.normal * prism_distance
        self.prism = Prism(prism_size=[30.,30.,30.], prism_center=prism_center)        

    def forward(self, undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1):
        ray_direct_1 = self.camera1.initialize_ray(undistorted_real_pixels_cam_0)
        ray_direct_2 = self.camera2.initialize_ray(undistorted_real_pixels_cam_1)
        self.prism.trace_ray(ray_direct_1.origin, ray_direct_1.origin + ray_direct_1.direction)
        self.prism.trace_ray(ray_direct_2.origin, ray_direct_2.origin + ray_direct_2.direction)
        recon_3D, _ = closest_point(ray_direct_1, ray_direct_2)
        return recon_3D

optical = True
if __name__=="__main__":
    if optical:
        n_glass = 1.55
        n_air = 1.
        prism_alpha = 0.
        prism_beta = pi / 2
        prism_gamma = pi / 2 
        prism_center = torch.tensor([0.,0.,0.])[:, None]
        
        prism = Prism(prism_size=[1.,1.,1.], prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                      prism_center=prism_center, refractive_index_glass=n_glass, 
                      refractive_index_air=n_air)
                
        origin_point=[0.,0.6,-0.25] 
        target_point=[0.,0.37,0.]
        prism.trace_ray(origin_point, target_point) 
        fig, ax = prism.visualize_prism_and_ray()
        ax.set_aspect('equal', adjustable='datalim')
        plt.show()
 
    else: 
        print('Not optical')
        prism_alpha = 0 * pi 
        prism_beta = 0.
        prism_gamma = pi / 2
        plane1 = Plane()
        plane2_center = torch.tensor([plane1.center[0,0] - plane1.a/2,
                                plane1.center[1,0],
                                plane1.center[2,0]])[:, None]
        plane2_b = plane1.b * torch.sqrt(torch.tensor(2))
        plane2 = Plane(beta=torch.pi/4, a = plane1.a, b=plane2_b, center=plane2_center)
        
        plane3_center = torch.array([plane1.center[0,0] - plane1.a/2, 
                                plane1.center[1,0],
                                plane1.center[2,0] - plane1.b/2])
        plane3 = Plane(beta=pi/2, a=plane1.a, b=plane1.b, center=plane3_center)

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
