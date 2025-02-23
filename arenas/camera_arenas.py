import numpy as np
import torch
import torch.nn as nn
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, EfficientCamera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
from utils import euclidean_distance, rotation_matrix_to_quaternion
pi = torch.tensor(np.pi, dtype=torch.float64)

class Arena_two_real_cameras_quat(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0,
    principal_point_pixel_cam_1,
    focal_length_cam_0,
    focal_length_cam_1,
    R_stereo_cam,
    T_stereo_cam,
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_two_real_cameras_quat, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = torch.tensor([stereo_alpha, stereo_beta, stereo_gamma]).to(torch.float64)
        self.stereo_camera_quat = nn.Parameter(rotation_matrix_to_quaternion(self.R_stereo_cam),
                                               requires_grad=True)
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     quat_stereo_cam,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(quat=quat_stereo_cam, T=T)        
        return camera2


    def forward(self, _, pixels_real_two_cams):
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        T2 = self.T_stereo_cam
        
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    self.stereo_camera_quat,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)                
        
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_real_pixels_cam_1 = pixels_real_two_cams[2:,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        recon_loss_real = (torch.norm(recon_distorted_real_pixels_cam_0 - distorted_real_pixels_cam_0, 
                    p=2, 
                    dim=0) + torch.norm(recon_distorted_real_pixels_cam_1 - distorted_real_pixels_cam_1, 
                    p=2, 
                    dim=0)) / 2
        
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_distorted_real_pixels_cam_1, recon_loss_real
    
    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                self.stereo_camera_quat,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])

        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)        
        ax.set_aspect('equal', adjustable='datalim')   

# Two cameras
class Arena_two_real_cameras(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_two_real_cameras, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                             dtype=torch.float64),
                                                requires_grad=False
                                                )
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2


    def forward(self, _, pixels_real_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R2 = R_stereo_cam
        T2 = self.T_stereo_cam
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)                
        
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_real_pixels_cam_1 = pixels_real_two_cams[2:,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        recon_loss_real = (torch.norm(recon_distorted_real_pixels_cam_0 - distorted_real_pixels_cam_0, 
                    p=2, 
                    dim=0) + torch.norm(recon_distorted_real_pixels_cam_1 - distorted_real_pixels_cam_1, 
                    p=2, 
                    dim=0)) / 2
        output = {}
        output['recon_3D'] = recon_3D
        output['closest_distance'] = closest_distance
        output['recon_distorted_real_pixels_cam_0'] = recon_distorted_real_pixels_cam_0
        output['recon_distorted_real_pixels_cam_1'] = recon_distorted_real_pixels_cam_1
        output['recon_loss_real'] = recon_loss_real
        return output    


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
        )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)        
        ax.set_aspect('equal', adjustable='datalim')   

#

    

# Two cameras grid distances
class Arena_two_real_cameras_grid_distance(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_two_real_cameras_grid_distance, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                             dtype=torch.float64),
                                                requires_grad=False
                                                )
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2


    def forward(self, pixels_real_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R2 = R_stereo_cam
        T2 = self.T_stereo_cam
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)                
        
        distorted_real_pixels_cam_0 = torch.hstack((pixels_real_two_cams[:2,:], 
                                                    pixels_real_two_cams[2:4,:]))
        distorted_real_pixels_cam_1 = torch.hstack((pixels_real_two_cams[4:6,:], 
                                                    pixels_real_two_cams[6:,:]))

        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0        
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        num_points = recon_3D.shape[1] // 2
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
                
        #recon_3D = recon_3D[:, :num_points]
        closest_distance = closest_distance[:num_points]
        #recon_distorted_real_pixels_cam_0 = recon_distorted_real_pixels_cam_0[:, :num_points]
        #recon_distorted_real_pixels_cam_1 = recon_distorted_real_pixels_cam_1[:, :num_points]
        recon_loss_real = (euclidean_distance(
            recon_distorted_real_pixels_cam_0,
            distorted_real_pixels_cam_0) + euclidean_distance(
            recon_distorted_real_pixels_cam_1,
            distorted_real_pixels_cam_1)) / 4
        output = {}
        output['recon_3D'] = recon_3D
        output['closest_distance'] = closest_distance
        output['recon_distorted_real_pixels_cam_0'] = recon_distorted_real_pixels_cam_0
        output['recon_distorted_real_pixels_cam_1'] = recon_distorted_real_pixels_cam_1
        output['recon_loss_real'] = recon_loss_real
        output['pairwise_distance'] = pairwise_distance
        return output


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[4:6, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]
        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
        )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)        
        ax.set_aspect('equal', adjustable='datalim')   