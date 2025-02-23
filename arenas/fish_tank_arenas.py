import numpy as np
import torch
import torch.nn as nn
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, EfficientCamera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
from utils import euclidean_distance, rotation_matrix_to_quaternion
pi = torch.tensor(np.pi, dtype=torch.float64)

class Arena_fish_tank(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    tank_thickness=None,
    tank_angles=None,
    tank_center=None,
    tank_size=None,
    refractive_index_acrylic=None,
    refractive_index_water=None):
        super(Arena_fish_tank, self).__init__()

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
        self.tank_center = nn.Parameter(tank_center)
        self.tank_angles = nn.Parameter(tank_angles)
        self.tank_thickness = nn.Parameter(tank_thickness)
        self.refractive_index_acrylic = nn.Parameter(refractive_index_acrylic)
        self.refractive_index_water = nn.Parameter(refractive_index_water)
        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.tank_size = nn.Parameter(tank_size)
        self.refractive_index_air = torch.ones_like(refractive_index_acrylic)
        

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

    def forward(self, pixels_two_cams):
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
        side_plane1, side_plane2, top_plane = self.get_tank_planes()
        pixels_side_cam = pixels_two_cams[:2, :]
        pixels_top_cam = pixels_two_cams[2:, :]
        pixels_side_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_side_cam, self.radial_dist_coeffs_cam_0)
        pixels_top_cam_undistorted = camera2.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_1)
        ray = camera2(pixels_side_cam_undistorted)
        ray2, _ = side_plane1(ray)
        ray_side_cam, _ = side_plane2(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, _ = top_plane(ray)
        recon_3D, closest_distance = closest_point(ray_side_cam, ray_top_cam)
        return recon_3D, closest_distance
    
    def visualize(self, pixels_two_cams, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_0 = pixels_two_cams[:2, :].clone()
        undistorted_pixels_cam_1 = pixels_two_cams[2:, :].clone()
        side_plane1, side_plane2, top_plane = self.get_tank_planes()

        test_idx = torch.randperm(undistorted_pixels_cam_0.shape[1])[:num_samples]
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
        ray1s = camera2.initialize_ray(undistorted_pixels_cam_0[:, test_idx])
        ray2s, _ = side_plane1(ray1s)
        ray_side, _ = side_plane2(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax, color_labels)
        fig, ax = side_plane2.visualize(fig, ax, color_labels)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_1[:, test_idx])
        ray_top, _ = top_plane(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim') 

    def get_tank_planes(self):
        tank_alpha, tank_beta, tank_gamma = self.tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        axes1 = torch.mm(rot_mat, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t()
                            )        
        
        side_plane1_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        )
        side_plane1 = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes1,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane1_center,
        )
        side_plane2_center = side_plane1_center - self.tank_thickness * axes1[:,0].unsqueeze(-1)


        side_plane2 = RefractingPlane(
            axes=axes1,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane2_center,
        )

        rot_mat_top = get_rot_mat(torch.tensor(0.).to(torch.float64),
                                  torch.tensor(pi/2).to(torch.float64),
                                  torch.tensor(0.).to(torch.float64))
        axes_top = torch.mm(rot_mat_top, 
                            axes1,
                            )
        axes_top = torch.mm(rot_mat_top,
                            axes_top,
                            )
        
        top_plane_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        ) - self.tank_size[0] / 2 * axes1[:,0].unsqueeze(-1) + self.tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[0],
            b=self.tank_size[1],
            center=top_plane_center
        )
        return side_plane1, side_plane2, top_plane 

class Arena_fish_tank_pairwise_distances(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    tank_thickness=None,
    tank_angles=None,
    tank_center=None,
    tank_size=None,
    refractive_index_acrylic=None,
    refractive_index_water=None):
        super(Arena_fish_tank_pairwise_distances, self).__init__()

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
                                                )
        self.tank_center = nn.Parameter(tank_center)
        self.tank_angles = nn.Parameter(tank_angles, requires_grad=False)
        self.tank_thickness = nn.Parameter(tank_thickness)
        self.refractive_index_acrylic = nn.Parameter(refractive_index_acrylic)
        self.refractive_index_water = nn.Parameter(refractive_index_water)
        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.tank_size = nn.Parameter(tank_size, requires_grad=True)
        self.refractive_index_air = torch.ones_like(refractive_index_acrylic)
        

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

    def forward(self, pixels_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)
        side_plane1, side_plane2, top_plane = self.get_tank_planes()
        pixels_top_cam = torch.hstack((
            pixels_two_cams[:2, :], 
            pixels_two_cams[2:4,:])
        )
        pixels_side_cam = torch.hstack((
            pixels_two_cams[4:6, :],
            pixels_two_cams[6:,:])
        )
        pixels_side_cam_undistorted = camera2.undistort_pixels_classical(pixels_side_cam, self.radial_dist_coeffs_cam_1)
        pixels_top_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_0)
        ray = camera2(pixels_side_cam_undistorted)
        ray2, intersection_penalty_1 = side_plane1(ray)
        ray_side_cam, intersection_penalty_2 = side_plane2(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, intersection_penalty_3 = top_plane(ray)
        recon_3D, closest_distance = closest_point(ray_side_cam, ray_top_cam)
        num_points = recon_3D.shape[1] // 2
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )
        zero_columns_top = (ray_top_cam.direction == 0).all(dim=0)
        zero_columns_side = (ray_side_cam.direction == 0).all(dim=0)
        num_bad_rays = torch.sum(zero_columns_side) + torch.sum(zero_columns_top)
        return recon_3D, closest_distance, pairwise_distance, intersection_penalty_1 + intersection_penalty_2 + intersection_penalty_3, num_bad_rays


    def visualize(self, pixels_two_cams, rand_sample, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_top = torch.hstack(
            (pixels_two_cams[:2, :].clone(), pixels_two_cams[2:4, :].clone())
        )
        undistorted_pixels_cam_side = torch.hstack(
            (pixels_two_cams[4:6, :].clone(),
             pixels_two_cams[6:,:])
        )
        side_plane1, side_plane2, top_plane = self.get_tank_planes()

        if rand_sample:
            test_idx = torch.randperm(undistorted_pixels_cam_top.shape[1])[:num_samples]
        else:
            test_idx = torch.arange(0,num_samples)
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
        ray1s = camera2.initialize_ray(undistorted_pixels_cam_side[:, test_idx])
        ray2s, _ = side_plane1(ray1s)
        ray_side, _ = side_plane2(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        ray_side.t = 500 * ray_side.t
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax, color_labels)
        fig, ax = side_plane2.visualize(fig, ax, color_labels)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_top[:, test_idx])
        ray_top, _ = top_plane(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        ray_top.t = 500 * ray_top.t
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim') 
        return camera2, side_plane1, side_plane2, top_plane, pixels_two_cams

    def get_tank_planes(self):
        tank_alpha, tank_beta, tank_gamma = self.tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        # NOTE: Do not change the axes definition here. Tank angles should be provided as input. THe tank angles can be derived from axes
        axes1 = torch.mm(rot_mat, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t()
                            )        
        
        side_plane1_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        )
        
        side_plane1 = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes1,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane1_center,
        )
        side_plane2_center = side_plane1_center - self.tank_thickness * axes1[:,0].unsqueeze(-1)


        side_plane2 = RefractingPlane(
            axes=axes1,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane2_center,
        )

        rot_mat_top = get_rot_mat(torch.tensor(0.).to(torch.float64),
                                  torch.tensor(-pi/2).to(torch.float64),
                                  torch.tensor(0.).to(torch.float64))
        axes_top = torch.mm(rot_mat_top, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t(),
                            )
        axes_top = torch.mm(rot_mat,
                            axes_top
                            )

        top_plane_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        ) - self.tank_size[0] / 2 * axes1[:,0].unsqueeze(-1) + self.tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[0],
            b=self.tank_size[1],
            center=top_plane_center
        )
        return side_plane1, side_plane2, top_plane 