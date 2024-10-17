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
import os
import scipy.io as sio
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
    return torch.stack([
    torch.tensor([1.0, 0.0, 0.0], device=angle.device),
    torch.stack([torch.tensor(0.0, device=angle.device), torch.cos(angle), -torch.sin(angle)]),
    torch.stack([torch.tensor(0.0, device=angle.device), torch.sin(angle), torch.cos(angle)])
    ])


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
    return torch.stack([
    torch.stack([torch.cos(angle), torch.tensor(0.0, device=angle.device), torch.sin(angle)]),
    torch.stack(
                [torch.tensor(0.0, device=angle.device), 
                torch.tensor(1.0, device=angle.device), 
                torch.tensor(0.0, device=angle.device)]),
    torch.stack([-torch.sin(angle), torch.tensor(0.0, device=angle.device), torch.cos(angle)])
    ])


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
    return torch.stack([
    torch.stack([torch.cos(angle), -torch.sin(angle), torch.tensor(0.0, device=angle.device)]),
    torch.stack([torch.sin(angle), torch.cos(angle), torch.tensor(0.0, device=angle.device)]),
    torch.stack(
                [torch.tensor(0.0, device=angle.device), 
                torch.tensor(0.0, device=angle.device), 
                torch.tensor(1.0, device=angle.device)])
    ])
    

def get_rot_mat(alpha, beta, gamma):
        return torch.mm(torch.mm(rotz(gamma), roty(beta)), rotx(alpha)) # Rotation matrix


# %%Ray class (for a ray of light)
class Ray():
    def __init__(self, origin=None, direction=None, target=None):
        """
        Parameters:
        - origin (2-D list): Origin of the ray.
        - direction (2-D list): Direction of the ray.
        """
        
        if (origin is not None) and (direction is not None):
            self.origin = origin
            direction = direction / torch.linalg.vector_norm(direction, dim=0).to(torch.float32)
            self.direction = direction

        if (origin is not None) and (target is not None):
            self.build_ray(origin=origin, target=target)

        if (origin is None) and (direction is None) and (target is None):
            if not isinstance(origin, torch.Tensor):
                origin = torch.tensor(origin, dtype=torch.float32, requires_grad=True)
                if len(origin.shape) == 1:
                    origin = origin.reshape((3, 1))
                if not isinstance(direction, torch.Tensor):
                    direction = torch.tensor(direction, dtype=torch.float32, requires_grad=True)
                    if len(direction.shape) == 1:
                        direction = direction.reshape((3, 1))
            self.origin = origin
            self.direction = direction
        self.t = torch.ones((self.direction.shape[1], 1), dtype=torch.float32, device=origin.device)

    def size(self):
        return self.direction.shape
    
    def build_ray(self, origin, target):
        """
        Build a ray from two points.
        Parameters:
        - point1 (2-D list): First point (this will become the origin of the ray).
        - point2 (2-D list): Second point.
        """
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=True)
            if len(origin.shape) == 1:
                origin = origin.reshape((3, 1))
        if not isinstance(target, torch.Tensor):
            point2 = torch.tensor(target, dtype=torch.float32, requires_grad=True)
            if len(target.shape) == 1:
                target = target.reshape((3, 1))

        self.origin = origin
        self.direction = (target - origin) / torch.linalg.vector_norm(target - origin, dim=0)
        self.t = torch.ones((self.direction.shape[1], 1), dtype=torch.float32, device=origin.device)
        
    def distance_to_point(self, point):
        """
        Get the distance of the ray to a point.
        Parameters:
        - point (2-D list): Point.
        Returns:
        - distance (2-D list): Distance of the ray to the point.
        """
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, dtype=torch.float32, requires_grad=True)
            if len(point.shape) == 1:
                point = point.reshape((3, 1))
        distance = torch.linalg.cross(self.direction, (point - self.origin), dim=0)
        distance = torch.linalg.norm(distance, dim=0)
        return distance

    def visualize(self, fig=None, ax=None, color_labels=None):
        """
        Visualize the ray.
        Parameters:
        - fig (object): Figure object.
        - ax (object): Axes object.
        - color_labels (bool): If True, color the rays using a colormap
        """
        num_rays = self.t.shape[0]
        if color_labels:
            colors = plt.get_cmap('jet')(np.linspace(0, 1.0, num_rays))

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
        for ray_id in range(num_rays):
            if color_labels:
                ray_color = colors[ray_id]
            else:
                ray_color = 'black'
            ax.plot([self.origin[0, ray_id].detach().numpy(),
             (self.origin[0, ray_id] + t[ray_id, 0] * self.direction[0, ray_id]).detach().numpy()],
             [self.origin[1, ray_id].detach().numpy(), (self.origin[1, ray_id] + t[ray_id, 0] * self.direction[1, ray_id]).detach().numpy()],
             [self.origin[2, ray_id].detach().numpy(), (self.origin[2, ray_id] + t[ray_id, 0] * self.direction[2, ray_id]).detach().numpy()],
             color=ray_color, linewidth=0.4)

        ax.quiver(self.origin[0].detach().numpy(), 
                  self.origin[1].detach().numpy(), 
                  self.origin[2].detach().numpy(), 
                  t * self.direction[0].detach().numpy(), 
                  t * self.direction[1].detach().numpy(), 
                  t * self.direction[2].detach().numpy(),
                  length=0.05, linewidth=0.1, color='black', normalize=True)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()
        return fig, ax


# %% Plane class
class Plane(nn.Module):
    def __init__(self, axes=None, center=None, alpha=0., beta=0., gamma=0., a=1., b=1.):
        super(Plane, self).__init__()
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
        - axes ((3,3) tensor): Axes of the plane
                                Each column represents one axies
                                The first column is the normal to the plane
                                The second column is the 'horizontal' direction (plane dimension is a)
                                The third column is the 'vertical' direction (plane dimension is b)

        """        

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)

        self.a = a
        self.b = b
        

        if center is None:
            center = torch.tensor([0., 0., 0.], dtype=torch.float32)

            
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        
        if len(center.shape) == 1:
            center = center.reshape((3, 1))                        
        self.center = center
        self.a = a
        self.b = b
        self.center = center        
        if axes is None:
            rot_mat =  get_rot_mat(alpha, beta, gamma) # Rotation matrix
            axes = torch.mm(
                rot_mat,
                torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
                )
        self.axes = axes


    @property
    def sides(self):
        return self.angles_to_sides(self.alpha, self.beta, self.gamma)

    @property
    def horizontal_direction(self):
        return self.get_horizontal_direction()
    
    @property
    def vertical_direction(self):
        return self.get_vertical_direction()
    
    @property
    def alpha(self):
        ref_axes = self.axes
        gamma = torch.arctan2(ref_axes[1,0], ref_axes[0,0])
        ref_axes1 = torch.mm(rotz(-gamma), ref_axes)
        beta = torch.arctan2(-ref_axes1[2,0], ref_axes1[0,0])
        ref_axes2 = torch.mm(roty(-beta), ref_axes1)
        return torch.arctan2(ref_axes2[2,1], ref_axes2[1,1])

    @property
    def beta(self):
        ref_axes = self.axes
        gamma = torch.arctan2(ref_axes[1,0], ref_axes[0,0])
        ref_axes1 = torch.mm(rotz(-gamma), ref_axes)
        return torch.arctan2(-ref_axes1[2,0], ref_axes1[0,0])
        
    @property
    def gamma(self):
        return torch.arctan2(self.axes[1,0], self.axes[0,0])        
    
    def get_horizontal_direction(self, original_horizontal_direction=None, rot_mat=None):       
        #NOTE: You need the original vertical direction in case the plane has been rotate
        """
        if not isinstance(original_horizontal_direction, torch.Tensor):
            original_horizontal_direction = torch.tensor([0., 0., -1.], dtype=torch.float32, device=self.center.device).unsqueeze(-1)
        if not isinstance(original_horizontal_direction, torch.Tensor):
            original_horizontal_direction = torch.tensor(original_horizontal_direction, dtype=torch.float32, device=self.center.device)
        if len(original_horizontal_direction.shape) == 1:
            original_horizontal_direction = original_horizontal_direction.unsqueeze(-1)
        #horizontal_direction = torch.tensor([0., 0., -1.], dtype=torch.float32).unsqueeze(-1)
        if rot_mat is None:
            #rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
            rot_mat =  get_rot_mat(self.alpha, self.beta, self.gamma)
        horizontal_direction = torch.mm(rot_mat, original_horizontal_direction)
        """
        return self.axes[:,1].unsqueeze(-1)


    def get_vertical_direction(self, original_vertical_direction=None, rot_mat=None):
        #NOTE: You need the original vertical direction in case the plane has been rotated
        """
        if original_vertical_direction is None:
            original_vertical_direction = torch.tensor([0., 1., 0.], dtype=torch.float32, device=self.alpha.device).unsqueeze(-1)
        if not isinstance(original_vertical_direction, torch.Tensor):
            original_vertical_direction = torch.tensor(original_vertical_direction, dtype=torch.float32, device=self.alpha.device)
        if len(original_vertical_direction.shape) == 1:
            original_vertical_direction = original_vertical_direction.unsqueeze(-1)
        #vertical_direction = torch.tensor([0., 1., 0.], dtype=torch.float32).unsqueeze(-1)
        if rot_mat is None:
            #rot_mat = rotz(self.gamma) @ roty(self.beta) @ rotx(self.alpha) # Rotation matrix        
            rot_mat =  get_rot_mat(self.alpha, self.beta, self.gamma)
        vertical_direction = torch.mm(rot_mat, original_vertical_direction)
        """
        return self.axes[:,2].unsqueeze(-1)


    def rotate_plane(self, alpha_rot=0., beta_rot=0., gamma_rot=0., rot_mat=None):
        if rot_mat is None:
            rot_mat =  get_rot_mat(alpha_rot, beta_rot, gamma_rot) # Rotation matrix
        self.axes = torch.mm(rot_mat, self.axes)
        self.center = torch.mm(rot_mat, self.center)


    """
    def update_angles_after_rotation(self):
        ref_axes = self.axes
        self.gamma = torch.arctan2(ref_axes[1,0], ref_axes[0,0])
        ref_axes1 = torch.mm(rotz(-self.gamma), ref_axes)
        self.beta = torch.arctan2(-ref_axes1[2,0], ref_axes1[0,0])
        ref_axes2 = torch.mm(roty(-self.beta), ref_axes1)
        self.alpha = torch.arctan2(ref_axes2[2,1], ref_axes2[1,1])
    """ 
        
    def move_plane(self, displacement):
        self.center = self.center + displacement


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

        side1 = torch.tensor([[0., self.a/2, self.b/2], [0., -self.a / 2, self.b / 2]], device=self.a.device).T 
        side2 = torch.tensor([[0., self.a/2, -self.b/2], [0., -self.a / 2, -self.b / 2]], device=self.a.device).T 
        side3 = torch.tensor([[0., self.a/2, self.b/2], [0., self.a/2, -self.b/2]], device=self.a.device).T 
        side4 = torch.tensor([[0., -self.a/2, self.b/2], [0., -self.a/2, -self.b/2]],device=self.a.device).T 

        if alpha is None:
            alpha = 0
        if beta is None:
            beta = 0
        if gamma is None:
            gamma = 0
        rot_mat =  get_rot_mat(alpha, beta, gamma) # Rotation matrix        
        side1 = self.center + torch.mm(rot_mat, side1)
        side2 = self.center + torch.mm(rot_mat, side2)
        side3 = self.center + torch.mm(rot_mat, side3)
        side4 = self.center + torch.mm(rot_mat, side4)
        return [side1, side2, side3, side4]

        

    def get_intersection(self, ray):
        """
        Get the intersection point of a ray with the plane.
        Parameters:
        - ray (np.array): Ray to intersect with the plane.
        Returns:
        - intersection (np.array): Intersection point.
        """

        ray_t = torch.mm((self.center - ray.origin).T, self.axes[:,0].unsqueeze(-1)) / torch.mm(ray.direction.T, self.axes[:,0].unsqueeze(-1))
        ray.t = ray_t.clone().detach()
        intersection = ray.origin + ray_t.t() * ray.direction
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
            rot_mat =  get_rot_mat(alpha, beta, gamma) # Rotation matrix

        side1 = torch.mm(rot_mat, side1)
        side2 = torch.mm(rot_mat, side2)
        side3 = torch.mm(rot_mat, side3)
        side4 = torch.mm(rot_mat, side4)
        return [side1, side2, side3, side4]
 
    def visualize(self, fig=None, ax=None, color=[1.,0.,0.]):
        """
        Visualize the plane by plotting the sides and 500 points lying on the plane.
        """
        rot_mat =  get_rot_mat(self.alpha, self.beta, self.gamma) # Rotation matrix  
        sampled_points = torch.rand(3, 500)
        sampled_points[0,:] = 0
        sampled_points[1,:] = sampled_points[1,:] * self.a - self.a / 2
        sampled_points[2,:] = sampled_points[2,:] * self.b - self.b / 2
        sampled_points = self.center + torch.mm(rot_mat, sampled_points)
        
        length_of_normal = 0.2 #cm
        normal = self.axes[:,0].unsqueeze(-1) * length_of_normal
        center = self.center
        normal_line = torch.hstack((center, center + normal))
        s1, s2, s3, s4 = self.sides

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot random points on the plane surface
        ax.scatter(sampled_points[0].detach().numpy(),
                    sampled_points[1].detach().numpy(), 
                    sampled_points[2].detach().numpy(),
                    c=color,
                    s=1,
                    alpha=0.25)

        # Plot sides of the plane
        ax.plot(s1[0].detach().numpy(), s1[1].detach().numpy(), s1[2].detach().numpy(), c='black')
        ax.plot(s2[0].detach().numpy(), s2[1].detach().numpy(), s2[2].detach().numpy(), c='black')
        ax.plot(s3[0].detach().numpy(), s3[1].detach().numpy(), s3[2].detach().numpy(), c='black')
        ax.plot(s4[0].detach().numpy(), s4[1].detach().numpy(), s4[2].detach().numpy(), c='black')
        
        # Plot normal to the plane through the center
        ax.plot(normal_line[0].detach().numpy(),
                normal_line[1].detach().numpy(), 
                normal_line[2].detach().numpy(), 
                c='black')
        ax.set_xlabel('X (mm)', fontsize=24)
        ax.set_ylabel('Y (mm)', fontsize=24)
        ax.set_zlabel('Z (mm)', fontsize=24)
        
        # Plot plane axes with the center as the origin
        for axis_id in range(3):
            if axis_id == 0:
                width = 3
                color='black'
            else:
                width = 1
                if axis_id == 1:
                    color='blue'
                else: 
                    color='green'
            axis_tip = self.center + self.axes[:,axis_id].unsqueeze(-1) * (self.a + self.b) / 4
            axis_tip = axis_tip 
            ax.plot([self.center[0].detach().numpy(), axis_tip[0].detach().numpy()],
                    [self.center[1].detach().numpy(), axis_tip[1].detach().numpy()],
                    [self.center[2].detach().numpy(), axis_tip[2].detach().numpy()],
                    linewidth=width,
                    color=color,
                    )

        return fig, ax

# %% Clas RefractingPlane

class RefractingPlane(Plane, nn.Module):
    def __init__(self, alpha=0., beta=0., gamma=0., center=[0.,0.,0.], 
                 axes=None, refractive_idx_1=1., refractive_idx_2=1.,
                  a=1., b=1.):                  
        super(RefractingPlane, self).__init__(axes=axes, center=center, alpha=alpha, beta=beta, gamma=gamma, a=a, b=b)
        """
        refractive_idx_1 is on the side facing the normal
        refractive_idx_2 is on the side facing away from the normal
        """

        if not isinstance(refractive_idx_1, torch.Tensor):
            refractive_idx_1 = torch.tensor(refractive_idx_1, dtype=torch.float32)
        if not isinstance(refractive_idx_2, torch.Tensor):
            refractive_idx_2 = torch.tensor(refractive_idx_2, dtype=torch.float32)

        self.refractive_idx_1 = refractive_idx_1
        self.refractive_idx_2 = refractive_idx_2       
     
                

    def forward(self, ray):
        """
        Refract a ray
        Parameters:
        - ray (np.array): Ray to refract
        - n1 (float): Refractive index of the first medium
        - n2 (float): Refractive index of the second medium
        Returns:
        - refracted_ray (np.array): Refracted ray
        Total internal reflection is handled by setting the direction of the refracted ray to 0
        """

        bad_rays_mask = ~(torch.linalg.vector_norm(ray.direction, dim=0) > 1e-1)
        if self.center.dtype != ray.origin.dtype:
            self.center = self.center.to(ray.origin.dtype)
            
        intersection = self.get_intersection(ray)
        
        refractive_idx_1 = self.refractive_idx_1
        refractive_idx_2 = self.refractive_idx_2
        
        #incoming_ray_vertical = mat1[0].apply(ray.direction)
        #r2 = np.arctan2(incoming_ray_vertical[1], incoming_ray_vertical[0])
        cosi = torch.mm(ray.direction.T, self.axes[:,0].unsqueeze(-1)).T #angle of incidence
        cosi = torch.clip(cosi, -1., 1.) # floating point errors can lead to cosi being slightly outside [-1, 1]

        normal_multiplier = torch.ones(cosi.shape)
            
        normal_multiplier[cosi < 0] = -1
        refractive_idx_1 = refractive_idx_1 * torch.ones(cosi.shape)
        refractive_idx_2 = refractive_idx_2 * torch.ones(cosi.shape)
        
        temp = refractive_idx_1.clone()
        refractive_idx_1[cosi > 0] = refractive_idx_2[cosi > 0]
        refractive_idx_2[cosi > 0] = temp[cosi > 0]
        cosi = torch.abs(cosi)
        sini = torch.sqrt(1 - cosi**2 + 1e-6)     
        ref_ratio = refractive_idx_1 / refractive_idx_2   
        sinr = sini * ref_ratio
        tir_mask = (1 - sinr**2) < 0. # Total internal reflection
        cosr = torch.sqrt(1 - (sinr * (~tir_mask))**2 + 1e-6) 
        
        bad_rays_mask = bad_rays_mask * tir_mask
        normal_component = cosr - ref_ratio * cosi
        incident_ray_component = ref_ratio
        refracted_ray_direction = normal_component * self.axes[:,0].unsqueeze(-1) * normal_multiplier + incident_ray_component * ray.direction
        refracted_ray_direction[:, bad_rays_mask[0]] = 0. * refracted_ray_direction[:, bad_rays_mask[0]]
        refracted_ray = Ray(origin=intersection, direction=refracted_ray_direction)
        return refracted_ray

#%% class ReflectingPlane
class ReflectingPlane(Plane, nn.Module):
    def __init__(self, alpha=0., beta=0., gamma=0., 
                center=[0.,0.,0.], axes=None,
                a=1., b=1.):
        super(ReflectingPlane, self).__init__(axes=axes, center=center, alpha=alpha, beta=beta, gamma=gamma, a=a, b=b) 
    
    def forward(self, ray):
        bad_rays_mask = ~(torch.linalg.vector_norm(ray.direction, dim=0) > 1e-1)
        if self.center.dtype != ray.origin.dtype:
            self.center = self.center.to(ray.origin.dtype)
        intersection = self.get_intersection(ray)
        #normal = self.angles_to_normal(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        cosi = torch.mm(ray.direction.T, self.axes[:,0].unsqueeze(-1)).T
        normal_multiplier = torch.ones(cosi.shape)
        normal_multiplier[cosi < 0] = -1
        cosi = torch.abs(cosi)

        displacement = 2 * self.axes[:,0].unsqueeze(-1) * normal_multiplier * cosi
        reflected_ray_direction = ray.direction - displacement
        reflected_ray_direction = reflected_ray_direction / torch.linalg.vector_norm(reflected_ray_direction, dim=0)
        reflected_ray_direction[:, bad_rays_mask[0]] = 0. * reflected_ray_direction[:, bad_rays_mask[0]]
        reflected_ray = Ray(origin=intersection, direction=reflected_ray_direction)
        return reflected_ray    
        

#%% Camera  class
class Camera(Plane, nn.Module):
    """
    Define a pinhole camera.
    Parameters:
    - aperture ((3,1) tensor): Aperture of the camera (0,0,0) for the primary camera
    - width (float): Width of the camera sensor in mm
    - height (float): Height of the camera sensor in mm
    - focal_length_pixels (float): Focal length of the camera in pixels
    - pixel_size (float): Pixel size in mm (assumes a square pixel)
    - principal_point_pixel ((2,1) tensor): Principal point (on the sensor) in pixels
    """
    # NOTE: The plane defines the camera sensor, not the aperture plane
    def __init__(self, alpha=0., beta=0., gamma=0., aperture=[0.,0.,0.], 
                 axes=None, height=4.9152, width=6.144, focal_length_pixels=5696.3, pixel_size=4.8e-3, 
                 principal_point_pixel=None):
        axes=torch.tensor([
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]])
        super(Camera, self).__init__(axes=axes, center=[0.,0.,0], alpha=alpha, beta=beta, gamma=gamma, a=width, b=height)
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
        self.focal_length_pixels = focal_length_pixels
        self.focal_length = self.focal_length_pixels * pixel_size
        self.pixel_size = pixel_size
        self.n_horizontal_pixels = width / self.pixel_size
        self.n_vertical_pixels = height / self.pixel_size
        self.principal_point = self.get_principal_point_from_aperture()
        self.principal_point_pixel = principal_point_pixel
        
        # Plane (defining the camera) center should be shifted so that the principal point is along the normal plane through the aperture
        delta_principal_point = torch.tensor([self.a/2 - self.pixel_size*principal_point_pixel[0], self.b/2 - self.pixel_size*principal_point_pixel[1]]) 
        self.center = ((self.principal_point[0,:] + delta_principal_point[0]) * self.horizontal_direction + (self.principal_point[1,:] + delta_principal_point[1]) * self.vertical_direction) + self.principal_point
                

    def forward(self, pixel, distortion_parameters=None):
        if not isinstance(pixel, torch.Tensor):
            pixel = torch.tensor(pixel, dtype=torch.float32)
            if len(pixel.shape) == 1:
                pixel = pixel.reshape((2, 1))
        r = torch.sqrt(torch.sum(pixel ** 2, dim=0))[None, :]
        if distortion_parameters:
            distortion_factor = (1 + distortion_parameters[0] * r**2 + distortion_parameters[1] ** r**4 + distortion_parameters[2] * r**6) / ...
            (1 + distortion_parameters[3] * r**2 + distortion_parameters[4] ** r**4 + distortion_parameters[5] * r**6)
            pixel = (pixel - self.principal_point_pixel) * distortion_factor + self.principal_point_pixel
        pixels = self.pixels_to_world(pixel)
        aperture = self.aperture.repeat(1, pixels.shape[1]).clone()
        ray = Ray(origin=pixels, target=aperture)
        return ray

    def reproject_using_intrinsics(self, world_coordinate):
        intrinsic_matrix = torch.tensor(
            [[self.focal_length_pixels, 0., self.principal_point_pixel[0], 0.], 
             [0, self.focal_length_pixels, self.principal_point_pixel[1], 0.], 
             [0., 0., 1., 0.]]
             )
        world_coordinate = torch.vstack((world_coordinate,
                                          torch.zeros(1,world_coordinate.shape[1])))
        reprojected_pixels_hom = intrinsic_matrix @ world_coordinate
        return reprojected_pixels_hom[:2, :] / reprojected_pixels_hom[2, :][None, :]


    def get_principal_point_from_aperture(self):
        return self.aperture - self.focal_length * self.axes[:,0].unsqueeze(-1)
        
    def initialize_ray(self, pixel):
        """
        Converts pixel coordinates to world coordinates and initializes a ray.
        """
        if not isinstance(pixel, torch.Tensor):
            pixel = torch.tensor(pixel, dtype=torch.float32)
            if len(pixel.shape) == 1:
                pixel = pixel.reshape((2, 1))
        pixels = self.pixels_to_world(pixel)
        aperture = self.aperture.repeat(1, pixels.shape[1]).clone()
        ray = Ray(origin=pixels, target=aperture)
        return ray
    
    def pixels_to_world(self, digital_pixels):
        """
        Convert pixel coordinates to world coordinates.
        pixels: (N,2) array of pixel coordinates
        """
        d_digital_pixels = digital_pixels - self.principal_point_pixel # Distance of the pixels from the principal point in the digital coordinates
        d_world_pixels = d_digital_pixels * self.pixel_size # Distance of the pixels from the principal point in the world coordinates
        # NOTE: Pixels are moved in the negative direction because the pinhole model inverts the image along both axes
        pixels = d_world_pixels[0,:] * (-1) * self.horizontal_direction + d_world_pixels[1,:] * (-1) * self.vertical_direction + self.principal_point # pixel location on the sensor in world coordinates
        return pixels
    
    def update_camera_pose(self, R, T):
        """
        Update the camera pose, given the camera extrinsics as Rotation Matrix (R) and Translation vector (t)
        """
        self.move_plane(displacement=self.aperture - self.center) # Move center to aperture
        self.move_plane(displacement=-T.clone())
        self.rotate_plane(rot_mat=R.clone()) 
        self.aperture = self.center
        self.center = self.aperture - self.focal_length * self.axes[:,0].unsqueeze(-1)
        self.center = self.center + torch.cat((
            self.principal_point_pixel[:,0] * self.pixel_size - torch.tensor([self.a/2, self.b/2]), torch.tensor([0.])))[:, None]
        self.principal_point = self.get_principal_point_from_aperture()
        

def visualize_camera_configuration(camera=None, prism=None, pixels=None, ax=None, fig=None, color_labels=None):
    """
    Visualize the camera configuration.
    """
    if camera is None:
        camera = Camera()
    
    prism_distance = 120.

    if prism is None:
        prism_center = camera.aperture + camera.axes[:,0].unsqueeze(-1) * prism_distance
        prism = Prism(prism_size=[20., 20., 20.], prism_center=prism_center, prism_angles=[pi,0.,0.])
            
    if pixels is None:
        pixels = torch.tensor([1,1]).unsqueeze(-1)
    
    ray = camera.initialize_ray(pixels)
    #prism(ray)
    fig, ax = camera.visualize(fig=fig, ax=ax)
    fig, ax = prism.visualize_prism_and_ray(ray, fig=fig, ax=ax, color_labels=color_labels)
    ax.scatter(camera.aperture[0].detach().numpy(), 
               camera.aperture[1].detach().numpy(), 
               camera.aperture[2].detach().numpy(), 
               c='black', s=10)
    ax.set_aspect('equal', adjustable='datalim')    
    ax.set_xlabel('X (mm)', fontsize=24)
    ax.set_ylabel('Y (mm)', fontsize=24)
    ax.set_zlabel('Z (mm)', fontsize=24)    
    return fig, ax, prism, camera



# %% Prism class
class Prism(nn.Module):

    def __init__(self, 
                prism_size=[1.,1.,1.], 
                prism_angles=[0.,0.,0.], 
                prism_center=[0.,0.,0.], 
                refractive_index_glass=1.5, 
                refractive_index_air=1.,
                adhesion_thickness_factor=20,
                refractive_index_adhesion=1.5):
        """
        Parameters:
        - prism_size (list): Width (X), Height (Y) and Depth (Z) of the prism.
        - prism_angles (list): A list of angles alpha (X-axis), beta (Y-axis), gamma (Z-axis)
        - prism_center (list): Center of the first surface of the prism. (surface facing the camera)
        """
        super(Prism, self).__init__()
        if not isinstance(prism_center, torch.Tensor):
            prism_center = torch.tensor(prism_center, 
                                        dtype=torch.float32)
            if len(prism_center.shape) == 1:
                prism_center = prism_center.reshape((3, 1))        

        if not isinstance(prism_angles, torch.Tensor):
            prism_angles = torch.tensor(prism_angles, 
                                        dtype=torch.float32)    
        
        if not isinstance(refractive_index_glass, torch.Tensor):
            refractive_index_glass = torch.tensor(
                                                refractive_index_glass,
                                                dtype=torch.float32)
        
        if not isinstance(prism_size, torch.Tensor):
         
            prism_size = torch.tensor(prism_size, 
                                        dtype=torch.float32)

        self.prism_size = prism_size
        self.prism_angles = prism_angles
        self.prism_center = prism_center
        self.refractive_index_glass = refractive_index_glass
        self.refractive_index_air = refractive_index_air
        self.adhesion_thickness_factor = adhesion_thickness_factor
        self.refractive_index_adhesion = refractive_index_adhesion
        
    def get_planes(self, prism_center, prism_angles):
       
        prism_alpha, prism_beta, prism_gamma = prism_angles
        rot_mat = get_rot_mat(prism_alpha, prism_beta, prism_gamma)
        axes1 = torch.mm(rot_mat, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float32).t()
                            )
        
        plane1 = RefractingPlane(
                            refractive_idx_1=self.refractive_index_air,
                            refractive_idx_2=self.refractive_index_glass,
                            axes=axes1,
                            a=self.prism_size[0], 
                            b=self.prism_size[1],
                            center=prism_center,
                            ) # Plane facing the camera
       
        rot_mat_135 = get_rot_mat(0.,3*pi/4,0.)
        axes_135 = torch.mm(rot_mat_135, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float32).t()
                            )
        
        axes2 = torch.mm(rot_mat, axes_135)

        
        plane2_center = plane1.center - plane1.axes[:,0].unsqueeze(-1) * self.prism_size[1] / 2
        plane2 = RefractingPlane(
                            refractive_idx_1=self.refractive_index_air,
                            refractive_idx_2=self.refractive_index_adhesion,
                            axes=axes2,
                            a=self.prism_size[0],
                            b=self.prism_size[1] * torch.sqrt(torch.tensor(2.)),
                            center=plane2_center,
                            )

        plane3_center = plane1.center - plane1.axes[:,0].unsqueeze(-1) * (self.prism_size[1] / 2 + self.prism_size[1] / self.adhesion_thickness_factor)
        plane3 = ReflectingPlane(
                            axes=axes2,
                            a=self.prism_size[0],
                            b=self.prism_size[1] * torch.sqrt(torch.tensor(2.)),
                            center=plane3_center,
                            )

        
        rot_90 = get_rot_mat(0.,pi/2,0.)
        axes3 = torch.mm(rot_90,
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float32).t()
                            )
        axes3 = torch.mm(rot_mat, axes3)

        plane4_center = plane2.center + plane1.axes[:,2].unsqueeze(-1) * self.prism_size[1] / 2        
        plane4 = RefractingPlane(
                            refractive_idx_1=self.refractive_index_glass,
                            refractive_idx_2=self.refractive_index_air,
                            axes=axes3,
                            a = self.prism_size[0],
                            b = self.prism_size[1],
                            center=plane4_center,
                            )
        return plane1, plane2, plane3, plane4
    
   
    def rotate_prism(self, alpha=0., beta=0., gamma=0.):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=torch.float32)
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=torch.float32)

        alpha, beta, gamma, center = Plane().get_parameters_after_rotation(alpha=alpha, 
                                                                        beta=beta, 
                                                                        gamma=gamma, 
                                                                        center=self.prism_center)
        self.prism_center = center
        self.prism_angles = torch.tensor([alpha, beta, gamma], device=self.prism_center.device)
    
    
    def move_prism(self, displacement):
        if not isinstance(displacement, torch.Tensor):
            displacement = torch.tensor(displacement, dtype=torch.float32)
        if (len(displacement.shape) == 1):
            displacement = displacement.unsqueeze(-1)        
        self.prism_center_add(displacement)


    def forward(self, incident_ray):
        plane1, plane2, plane3, plane4 = self.get_planes(self.prism_center, self.prism_angles)
        ray1 = plane1(incident_ray)
        ray2 = plane2(ray1)
        ray3 = plane3(ray2)
        ray4 = plane4(ray3)
        return ray1, ray2, ray3, ray4
    
    def visualize_prism(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        plane1, plane2, plane3, plane4 = self.get_planes(self.prism_center,
                                                 self.prism_angles)
        fig, ax = plane1.visualize(fig, ax)
        fig, ax = plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        fig, ax = plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = plane4.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        ax.set_xlabel('X (mm)', fontsize=24)
        ax.set_ylabel('Y (mm)', fontsize=24)
        ax.set_zlabel('Z (mm)', fontsize=24)
        return fig, ax
    
    def visualize_prism_and_ray(self, incident_ray, fig=None, ax=None, color_labels=None):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        plane1, plane2, plane3, plane4= self.get_planes(self.prism_center,
                                                self.prism_angles)
        ray2 = plane1(incident_ray)
        ray3 = plane2(ray2)
        ray4 = plane3(ray3)
        ray5 = plane4(ray4)
        fig, ax = plane1.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = plane2.visualize(fig, ax, color=[[0.5, 0.5, 0.5]])
        fig, ax = plane3.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = plane4.visualize(fig, ax, color=[0.5, 0.5, 0.5])
        fig, ax = incident_ray.visualize(fig, ax, color_labels=color_labels)
        fig, ax = ray2.visualize(fig, ax, color_labels=color_labels)
        fig, ax = ray3.visualize(fig, ax, color_labels=color_labels)
        fig, ax = ray4.visualize(fig, ax, color_labels=color_labels)
        fig, ax = ray5.visualize(fig, ax, color_labels=color_labels)
        ax.set_xlabel('X (mm)', fontsize=24)
        ax.set_ylabel('Y (mm)', fontsize=24)
        ax.set_zlabel('Z (mm)', fontsize=24)
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


# %% Arena class with adhesion layer
class Arena(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance,
    prism_angles):
        super(Arena, self).__init__()
        # Camera initialization        
        self.camera1 = Camera(
            principal_point_pixel=principal_point_pixel_cam_0, 
            focal_length_pixels=focal_length_cam_0)
        self.camera2 = Camera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1)
        self.camera2.update_camera_pose(R, T)
        
        # Prism initialization
        prism_center = self.camera1.aperture + self.camera1.axes[:,0].unsqueeze(-1) * prism_distance
        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_angles = nn.Parameter(torch.tensor(prism_angles), requires_grad=True)
        refractive_index_glass = nn.Parameter(torch.tensor(1.5), requires_grad=True)
        self.prism = Prism(
            refractive_index_glass=refractive_index_glass,
            prism_size=[30.,30.,30.], 
            prism_center=prism_center, 
            prism_angles=prism_angles)        

    def forward(self, 
                undistorted_real_pixels_cam_0, 
                undistorted_real_pixels_cam_1):
        cam_1_ray = self.camera1.initialize_ray(
            undistorted_real_pixels_cam_0)
        cam_2_ray = self.camera2.initialize_ray(
            undistorted_real_pixels_cam_1)
        _, _, emergent_ray_1 = self.prism(cam_1_ray)
        _, _, emergent_ray_2 = self.prism(cam_2_ray)
        recon_3D, _ = closest_point(emergent_ray_1, emergent_ray_2)
        return recon_3D

optical = True
if __name__=="__main__":
    if optical:
        n_glass = 1.55
        n_air = 1.
        prism_alpha = 0.
        prism_beta = 0
        prism_gamma = 0
        prism_center = torch.tensor([0.,0.,0.])[:, None]
        
        prism = Prism(prism_size=[1.,1.,1.], 
                      prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                      prism_center=prism_center, 
                      refractive_index_glass=n_glass, 
                      refractive_index_air=n_air)
        
        npts = 25
        origin_point = torch.zeros(3, npts)
        target_point = torch.zeros(3, npts)

        origin_point[1,:] = torch.linspace(-0.35, 0.35, npts)
        origin_point[2,:] = -0.3
        target_point[1,:] = 0.075
        ray = Ray(origin=origin_point, target=target_point)
        _, _, _, emergent_ray = prism(ray) 
        fig, ax = prism.visualize_prism_and_ray(ray, color_labels=True)
        ax.set_aspect('equal', adjustable='datalim')
        plt.show()

    else:
        calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
        calibration_results_file = 'ball_bearing_data.mat'
        calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
        mat = sio.loadmat(calibration_results_path)
        undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_undistorted'], dtype=torch.float32, requires_grad=True).T - 1
        undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_undistorted'], dtype=torch.float32, requires_grad=True).T - 1
        target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T

        principal_point_pixel_cam_0 = [638.040 - 1, 492.499 - 1] # This comes from the calibration results
        principal_point_pixel_cam_1 = [659.3778 - 1, 521.5078 - 1]

        R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                        [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                        [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T
        T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None]
        focal_length_cam_1 = 19.65
        focal_length_cam_2 = 19.69
        prism_angles = [0., 0., 0.]
        prism_distance = torch.tensor(130.)
        arena = Arena(principal_point_pixel_cam_0, 
                    principal_point_pixel_cam_1, 
                    focal_length_cam_1, 
                    focal_length_cam_2, 
                    R, 
                    T, 
                    prism_distance,
                    prism_angles)
        pixels_two_cams = [undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1]
        arena(undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1)

# %%
