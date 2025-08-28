import numpy as np
import torch
import torch.nn as nn
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, EfficientCamera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
from utils import euclidean_distance, rotation_matrix_to_quaternion
pi = torch.tensor(np.pi, dtype=torch.float64)
from ray_tracing_simulator_nnModules_grad import rotx, roty, rotz

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
        side_plane1, side_plane2_outer, top_plane_outer = self.get_tank_planes()
        pixels_side_cam = pixels_two_cams[:2, :]
        pixels_top_cam = pixels_two_cams[2:, :]
        pixels_side_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_side_cam, self.radial_dist_coeffs_cam_0)
        pixels_top_cam_undistorted = camera2.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_1)
        ray = camera2(pixels_side_cam_undistorted)
        ray2, _ = side_plane1(ray)
        ray_side_cam, _ = side_plane2_outer(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, _ = top_plane_outer(ray)
        recon_3D, closest_distance = closest_point(ray_side_cam, ray_top_cam)
        return recon_3D, closest_distance
    
    def visualize(self, pixels_two_cams, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_0 = pixels_two_cams[:2, :].clone()
        undistorted_pixels_cam_1 = pixels_two_cams[2:, :].clone()
        side_plane1, side_plane2_outer, top_plane_outer = self.get_tank_planes()

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
        ray_side, _ = side_plane2_outer(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax, color_labels)
        fig, ax = side_plane2_outer.visualize(fig, ax, color_labels)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_1[:, test_idx])
        ray_top, _ = top_plane_outer(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane_outer.visualize(fig, ax, color_labels)
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
        side_plane2_outer_center = side_plane1_center - self.tank_thickness * axes1[:,0].unsqueeze(-1)


        side_plane2_outer = RefractingPlane(
            axes=axes1,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane2_outer_center,
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
        
        top_plane_outer_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        ) - self.tank_size[0] / 2 * axes1[:,0].unsqueeze(-1) + self.tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane_outer = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[0],
            b=self.tank_size[1],
            center=top_plane_outer_center
        )
        return side_plane1, side_plane2_outer, top_plane_outer 

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
        side_plane1, side_plane2_outer, top_plane_outer = self.get_tank_planes()
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
        ray_side_cam, intersection_penalty_2 = side_plane2_outer(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, intersection_penalty_3 = top_plane_outer(ray)
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
        side_plane1, side_plane2_outer, top_plane_outer = self.get_tank_planes()

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
        ray_side, _ = side_plane2_outer(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        ray_side.t = 500 * ray_side.t
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax)
        fig, ax = side_plane2_outer.visualize(fig, ax)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_top[:, test_idx])
        ray_top, _ = top_plane_outer(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        ray_top.t = 500 * ray_top.t
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane_outer.visualize(fig, ax)
        ax.set_aspect('equal', adjustable='datalim') 
        return camera2, side_plane1, side_plane2_outer, top_plane_outer, pixels_two_cams

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
    

class Arena_Akihiro_fish_tank_pairwise_distances(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    principal_point_pixel_cam_2,
    focal_length_cam_0, 
    focal_length_cam_1,
    focal_length_cam_2, 
    R_stereo_cam1, 
    T_stereo_cam1, 
    R_stereo_cam2,
    T_stereo_cam2,
    outer_tank_thickness=None,
    outer_tank_angles=None,
    outer_tank_center=None,
    outer_tank_size=None,
    inner_tank_thickness=None,
    inner_tank_angles=None,
    inner_tank_distance=None,
    inner_tank_size=None,
    refractive_index_acrylic=None,
    refractive_index_water=None):
        super(Arena_Akihiro_fish_tank_pairwise_distances, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_2 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_2, dtype=torch.float64).reshape(2,1),
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

        focal_length_cam_2 = nn.Parameter(
            torch.tensor(focal_length_cam_2, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam1_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam2_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.principal_point_pixel_cam_2 = principal_point_pixel_cam_2
        self.focal_length_cam_1 = focal_length_cam_1
        self.focal_length_cam_2 = focal_length_cam_2
        self.R_stereo_cam1 = R_stereo_cam1
        self.T_stereo_cam1 = nn.Parameter(T_stereo_cam1, requires_grad=True)
        self.R_stereo_cam2 = R_stereo_cam2
        self.T_stereo_cam2 = nn.Parameter(T_stereo_cam2, requires_grad=True)
        stereo_alpha1, stereo_beta1, stereo_gamma1 = self.get_stereo_camera_angles(R_stereo_cam1)
        stereo_alpha2, stereo_beta2, stereo_gamma2 = self.get_stereo_camera_angles(R_stereo_cam2)
        self.stereo_camera1_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha1, stereo_beta1, stereo_gamma1], 
                                                             dtype=torch.float64),
                                                )
        self.stereo_camera2_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha2, stereo_beta2, stereo_gamma2], 
                                                             dtype=torch.float64),
                                                )
        self.outer_tank_center = nn.Parameter(outer_tank_center)
        self.outer_tank_angles = nn.Parameter(outer_tank_angles, requires_grad=False)
        self.inner_tank_angles = outer_tank_angles
        self.outer_tank_thickness = nn.Parameter(outer_tank_thickness)
        self.inner_tank_distance = nn.Parameter(inner_tank_distance)
        self.inner_tank_angles = nn.Parameter(inner_tank_angles, requires_grad=False)
        self.inner_tank_thickness = nn.Parameter(inner_tank_thickness)
        self.refractive_index_acrylic = nn.Parameter(refractive_index_acrylic)
        self.refractive_index_water = nn.Parameter(refractive_index_water)
        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_2 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.inner_tank_size = nn.Parameter(inner_tank_size, requires_grad=True)
        self.outer_tank_size = nn.Parameter(outer_tank_size, requires_grad=True)        
        self.refractive_index_air = torch.ones_like(refractive_index_acrylic)
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    
    def get_stereo_camera(self, 
                     principal_point_pixel_cam,
                     focal_length_cam,
                     R,
                     T,
                     r1):
        camera = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam, 
            focal_length_pixels=focal_length_cam, 
            r1=r1)
        camera.update_camera_pose(R, T)
        return camera

    def forward(self, pixels_two_cams):
        R_stereo_cam1 = get_rot_mat(
            self.stereo_camera1_angles[0],
            self.stereo_camera1_angles[1],
            self.stereo_camera1_angles[2],
            )
        # Side camera 1
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam1,
                                    self.T_stereo_cam1,
                                    r1=self.stereocam1_r1)
        R_stereo_cam2 = get_rot_mat(
            self.stereo_camera2_angles[0],
            self.stereo_camera2_angles[1],
            self.stereo_camera2_angles[2],
            )
        
        # Side camera 2
        camera3 = self.get_stereo_camera(self.principal_point_pixel_cam_2,
                                    self.focal_length_cam_2,
                                    R_stereo_cam2,
                                    self.T_stereo_cam2,
                                    r1=self.stereocam2_r1)
        
        side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, top_plane1_outer, top_plane2_outer = self.get_outer_tank_planes()
        side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_inner, top_plane2_inner = self.get_inner_tank_planes()
    
        # NOTE: The order of the planes is important. The outer plane is the one that is hit first.
        pixels_top_cam = torch.hstack((
            pixels_two_cams[:2, :], 
            pixels_two_cams[2:4,:])
        )
        pixels_side_cam1 = torch.hstack((
            pixels_two_cams[4:6, :],
            pixels_two_cams[6:8,:])
        )
        pixels_side_cam2 = torch.hstack((
            pixels_two_cams[8:10, :],
            pixels_two_cams[10:12,:])
        )
        pixels_side_cam1_undistorted = camera2.undistort_pixels_classical(pixels_side_cam1, self.radial_dist_coeffs_cam_1)
        pixels_side_cam2_undistorted = camera3.undistort_pixels_classical(pixels_side_cam2, self.radial_dist_coeffs_cam_2)
        pixels_top_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_0)
        
        # Ray tracing from camera 1 (top camera)
        ray_temp = self.camera1(pixels_top_cam_undistorted)
        ray_temp, _ = top_plane1_outer(ray_temp)
        ray_top_cam, intersection_penalty_5 = top_plane2_outer(ray_temp)
        
        """
        NOTE: The inner planes are not used in this version of the code
        ray_temp, _ = top_plane1_inner(ray_top_cam)
        ray_top_cam, intersection_penalty_6 = top_plane2_inner(ray_temp)
        """

        intersection_penalty_6 = torch.zeros_like(intersection_penalty_5) # Until there is no inner top tank

        # Ray tracing from camera 2 (first side camera)
        ray_temp = camera2(pixels_side_cam1_undistorted)
        ray_temp, intersection_penalty_1 = side_plane11_outer(ray_temp)
        ray_side_cam1, intersection_penalty_2 = side_plane12_outer(ray_temp)
        ray_temp, intersection_penalty_1 = side_plane11_inner(ray_side_cam1)
        ray_side_cam1, intersection_penalty_2 = side_plane12_inner(ray_temp)
        
        # Ray tracing from camera 3 (second side camera)
        ray_temp = camera3(pixels_side_cam2_undistorted)
        ray_temp, intersection_penalty_3 = side_plane21_outer(ray_temp)
        ray_side_cam2, intersection_penalty_4 = side_plane22_outer(ray_temp)
        ray_temp, intersection_penalty_3 = side_plane21_inner(ray_side_cam2)
        ray_side_cam2, intersection_penalty_4 = side_plane22_inner(ray_temp)        

        recon_3D_top_side1, closest_distance_top_side1 = closest_point(ray_side_cam1, ray_top_cam)
        num_points = recon_3D_top_side1.shape[1] // 2
        pairwise_distance_1 = euclidean_distance(
            recon_3D_top_side1[:, :num_points],
            recon_3D_top_side1[:, num_points:]
        )

        recon_3D_top_side2, closest_distance_top_side2 = closest_point(ray_side_cam2, ray_top_cam)
        num_points = recon_3D_top_side2.shape[1] // 2
        pairwise_distance_2 = euclidean_distance(
            recon_3D_top_side2[:, :num_points],
            recon_3D_top_side2[:, num_points:]
        )
        
        recon_3D_side1_side2, closest_distance_side1_side2 = closest_point(ray_side_cam1, ray_side_cam2)
        num_points = recon_3D_side1_side2.shape[1] // 2
        pairwise_distance_3 = euclidean_distance(
            recon_3D_side1_side2[:, :num_points],
            recon_3D_side1_side2[:, num_points:]
        )

        recon_3D = torch.mean(
            torch.stack((recon_3D_top_side1, recon_3D_top_side2, recon_3D_side1_side2), dim=0),
            dim=0,
        )

        closest_distance = torch.mean(
            torch.stack((closest_distance_top_side1, closest_distance_top_side2, closest_distance_side1_side2), dim=0),
            dim=0,
        )

        pairwise_distance = torch.mean(
            torch.stack((pairwise_distance_1, pairwise_distance_2, pairwise_distance_3), dim=0),
            dim=0
        )

        zero_columns_top = (ray_top_cam.direction == 0).all(dim=0) # Used to calculate the number of "bad" rays
        zero_columns_side1 = (ray_side_cam1.direction == 0).all(dim=0)
        zero_columns_side2 = (ray_side_cam2.direction == 0).all(dim=0)
        num_bad_rays = torch.sum(zero_columns_side1) + torch.sum(zero_columns_top) + torch.sum(zero_columns_side2)
        return recon_3D, closest_distance, pairwise_distance, intersection_penalty_1 + intersection_penalty_2 + intersection_penalty_3 + intersection_penalty_4 + intersection_penalty_5, num_bad_rays


    def project_virtual_rays(self, pixels_two_cams):
        R_stereo_cam1 = get_rot_mat(
            self.stereo_camera1_angles[0],
            self.stereo_camera1_angles[1],
            self.stereo_camera1_angles[2],
            )
        # Side camera 1
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam1,
                                    self.T_stereo_cam1,
                                    r1=self.stereocam1_r1)
        R_stereo_cam2 = get_rot_mat(
            self.stereo_camera2_angles[0],
            self.stereo_camera2_angles[1],
            self.stereo_camera2_angles[2],
            )
        
        # Side camera 2
        camera3 = self.get_stereo_camera(self.principal_point_pixel_cam_2,
                                    self.focal_length_cam_2,
                                    R_stereo_cam2,
                                    self.T_stereo_cam2,
                                    r1=self.stereocam2_r1)
        
        side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, top_plane1_outer, top_plane2_outer = self.get_outer_tank_planes()
        side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_inner, top_plane2_inner = self.get_inner_tank_planes()
    
        # NOTE: The order of the planes is important. The outer plane is the one that is hit first.
        pixels_top_cam = pixels_two_cams[:2, :]
        pixels_side_cam1 = pixels_two_cams[2:4, :]
        pixels_side_cam2 = pixels_two_cams[4:6, :]
            
        pixels_side_cam1_undistorted = camera2.undistort_pixels_classical(pixels_side_cam1, self.radial_dist_coeffs_cam_1)
        pixels_side_cam2_undistorted = camera3.undistort_pixels_classical(pixels_side_cam2, self.radial_dist_coeffs_cam_2)
        pixels_top_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_0)
        
        # Ray tracing from camera 1 (top camera)
        ray_temp = self.camera1(pixels_top_cam_undistorted)
        ray_temp, _ = top_plane1_outer(ray_temp)
        ray_top_cam, intersection_penalty_5 = top_plane2_outer(ray_temp)
        """
        NOTE: The inner planes are not used in this version of the code
        ray_temp, _ = top_plane1_inner(ray_top_cam)
        ray_top_cam, intersection_penalty_6 = top_plane2_inner(ray_temp)
        """

        intersection_penalty_6 = torch.zeros_like(intersection_penalty_5) # Until there is no inner top tank

        # Ray tracing from camera 2 (first side camera)
        ray_temp = camera2(pixels_side_cam1_undistorted)
        ray_temp, intersection_penalty_1 = side_plane11_outer(ray_temp)
        ray_side_cam1, intersection_penalty_2 = side_plane12_outer(ray_temp)
        ray_temp, intersection_penalty_1 = side_plane11_inner(ray_side_cam1)
        ray_side_cam1, intersection_penalty_2 = side_plane12_inner(ray_temp)
        
        # Ray tracing from camera 3 (second side camera)
        ray_temp = camera3(pixels_side_cam2_undistorted)
        ray_temp, intersection_penalty_3 = side_plane21_outer(ray_temp)
        ray_side_cam2, intersection_penalty_4 = side_plane22_outer(ray_temp)
        ray_temp, intersection_penalty_3 = side_plane21_inner(ray_side_cam2)
        ray_side_cam2, intersection_penalty_4 = side_plane22_inner(ray_temp)        


        outputs = {}
        outputs['ray_side_cam_1'] = ray_side_cam1
        outputs['ray_side_cam_2'] = ray_side_cam2
        outputs['ray_top_cam'] = ray_top_cam
        return outputs

    def visualize(self, pixels_all_cams, rand_sample=True, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_top = torch.hstack(
            (pixels_all_cams[:2, :].clone(), pixels_all_cams[2:4, :].clone())
        )
        undistorted_pixels_cam_side1 = torch.hstack(
            (pixels_all_cams[4:6, :].clone(),
             pixels_all_cams[6:8,:])
        )
        undistorted_pixels_cam_side2 = torch.hstack(
            (pixels_all_cams[8:10, :].clone(),
             pixels_all_cams[10:12,:])
        )

        side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, top_plane1_outer, top_plane2_outer = self.get_outer_tank_planes()
        side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_inner, top_plane2_inner = self.get_inner_tank_planes()

        if rand_sample:
            test_idx = torch.randperm(undistorted_pixels_cam_top.shape[1])[:num_samples]
        else:
            test_idx = torch.arange(0,num_samples)
        R_stereo_cam1 = get_rot_mat(
            self.stereo_camera1_angles[0],
            self.stereo_camera1_angles[1],
            self.stereo_camera1_angles[2],
        )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam1,
                                self.T_stereo_cam1,
                                self.stereocam1_r1)
        R_stereo_cam2 = get_rot_mat(
            self.stereo_camera2_angles[0],
            self.stereo_camera2_angles[1],
            self.stereo_camera2_angles[2],
        )
        camera3 = self.get_stereo_camera(self.principal_point_pixel_cam_2,
                                self.focal_length_cam_2,
                                R_stereo_cam2,
                                self.T_stereo_cam2,
                                self.stereocam2_r1)
        
        # Outer tank, camera 2        
        fig, ax = camera2.visualize()
        ray1s = camera2.initialize_ray(undistorted_pixels_cam_side1[:, test_idx])        
        ray2s, _ = side_plane11_outer(ray1s)
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = side_plane11_outer.visualize(fig, ax)
        ray_side1, _ = side_plane12_outer(ray2s)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = side_plane12_outer.visualize(fig, ax)

        # Inner tank, camera 2
        ray2s, _ = side_plane11_inner(ray_side1)
        fig, ax = ray_side1.visualize(fig, ax, color_labels)
        fig, ax = side_plane11_inner.visualize(fig, ax)        
        ray_side1, _ = side_plane12_inner(ray2s)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = side_plane12_inner.visualize(fig, ax)
        ray_side1.t = 500 * ray_side1.t
        fig, ax = ray_side1.visualize(fig, ax, color_labels)

        # Outer tank, camera 3
        fig, ax = camera3.visualize(fig, ax)
        ray1s = camera3.initialize_ray(undistorted_pixels_cam_side2[:, test_idx])        
        ray2s, _ = side_plane21_outer(ray1s)
        fig, ax = ray1s.visualize(fig, ax, color_labels)        
        fig, ax = side_plane21_outer.visualize(fig, ax)
        ray_side1, _ = side_plane22_outer(ray2s)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = side_plane22_outer.visualize(fig, ax)                
        
        # Inner tank, camera 3
        fig, ax = self.camera1.visualize(fig, ax)        
        ray2s, _ = side_plane21_inner(ray_side1)
        fig, ax = ray_side1.visualize(fig, ax, color_labels)
        fig, ax = side_plane21_inner.visualize(fig, ax)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        ray_side1, _ = side_plane22_inner(ray2s)
        fig, ax = side_plane22_inner.visualize(fig, ax)
        fig, ax = ray_side1.visualize(fig, ax, color_labels)        
        ray_side1.t = 500 * ray_side1.t
        fig, ax = ray_side1.visualize(fig, ax, color_labels)                                                        

        # Top tank, camera 1
        fig, ax = self.camera1.visualize(fig, ax)
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_top[:, test_idx])
        ray2t, _ = top_plane1_outer(ray1t)
        fig, ax = ray1t.visualize(fig, ax, color_labels)          
        ray_top, _ = top_plane2_outer(ray2t)
        fig, ax = ray2t.visualize(fig, ax, color_labels)
        ray_top.t = 500 * ray_top.t
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane2_outer.visualize(fig, ax)
        fig, ax = top_plane1_outer.visualize(fig, ax)
        
        # Inner tank, camera 1
        """"
        ray2t, _ = top_plane1_inner(ray_top)
        fig, ax = top_plane1_inner.visualize(fig, ax)        
        ray_top, _ = top_plane2_inner(ray2t)
        fig, ax = ray2t.visualize(fig, ax, color_labels)
        ray_top.t = 500 * ray_top.t
        """
        fig, ax = ray_top.visualize(fig, ax, color_labels)        

        ax.set_aspect('equal', adjustable='datalim') 

        return camera2, camera3, side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_outer, top_plane2_outer, top_plane1_inner, top_plane2_inner, pixels_all_cams, 

    def get_outer_tank_planes(self):
        # side_plane11 (facing the camera), side_plane12, side_plane21 (facing the camera), side_plane22
        tank_alpha, tank_beta, tank_gamma = self.outer_tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        # NOTE: Do not change the axes definition here. Tank angles should be provided as input. THe tank angles can be derived from axes
        axes11 = torch.mm(rot_mat, 
                            torch.tensor(
                                [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], 
                                dtype=torch.float64).t()
                            )        
        
        side_plane11_outer_center = torch.vstack([
            self.outer_tank_center[0],
            self.outer_tank_center[1],
            self.outer_tank_center[2],
        ]
        )        
        side_plane11_outer = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes11,
            a=self.outer_tank_size[1],
            b=self.outer_tank_size[2],
            center=side_plane11_outer_center,
        )

        side_plane12_outer_center = side_plane11_outer_center - self.outer_tank_thickness * axes11[:,0].unsqueeze(-1)
        side_plane12_outer = RefractingPlane(
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            axes=axes11, # Same axis as plane11
            a=self.outer_tank_size[1],
            b=self.outer_tank_size[2],
            center=side_plane12_outer_center,
        )

        #NOTE: This is hard-coded for now, but needs to be changed later
        axes21 = torch.mm(rot_mat @ rotz(-pi / 2), 
                            torch.tensor(
                                [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], 
                                dtype=torch.float64).t()
                            )        
        
        side_plane21_outer_center = self.outer_tank_center.unsqueeze(-1) + self.outer_tank_size[0] / 2 * axes21[:,0].unsqueeze(-1) - self.outer_tank_size[1] / 2 * axes21[:,1].unsqueeze(-1)        
        side_plane21_outer = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes21,
            a=self.outer_tank_size[1],
            b=self.outer_tank_size[2],
            center=side_plane21_outer_center,
        )

        # Making side_plane_22 parallel to side_plane21 (wall thickness)        
        side_plane22_outer_center = side_plane21_outer_center - self.outer_tank_thickness * axes21[:,0].unsqueeze(-1)
        side_plane22_outer = RefractingPlane(
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            axes=axes21,
            a=self.outer_tank_size[1],
            b=self.outer_tank_size[2],
            center=side_plane22_outer_center,
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

        top_plane1_outer_center = torch.vstack([
            self.outer_tank_center[0],
            self.outer_tank_center[1],
            self.outer_tank_center[2],
        ]
        ) - self.outer_tank_size[0] / 2 * axes11[:,0].unsqueeze(-1) + self.outer_tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane1_outer = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            a=self.outer_tank_size[0],
            b=self.outer_tank_size[1],
            center=top_plane1_outer_center
        )

        top_plane2_outer_center = top_plane1_outer_center - self.outer_tank_thickness * axes_top[:,0].unsqueeze(-1)
        top_plane2_outer = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.outer_tank_size[0],
            b=self.outer_tank_size[1],
            center=top_plane2_outer_center
        )
        return side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, top_plane1_outer, top_plane2_outer
    
    
    def get_inner_tank_planes(self):
        # side_plane11 (facing the camera), side_plane12, side_plane21 (facing the camera), side_plane22
        tank_alpha, tank_beta, tank_gamma = self.inner_tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        # NOTE: Do not change the axes definition here. Tank angles should be provided as input. THe tank angles can be derived from axes
        axes11 = torch.mm(rot_mat, 
                            torch.tensor(
                                [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], 
                                dtype=torch.float64).t()
                            )        
        #NOTE: This is hard-coded for now, but needs to be changed later
        axes21 = torch.mm(rot_mat @ rotz(-pi / 2), 
                            torch.tensor(
                                [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], 
                                dtype=torch.float64).t()
                            )        
        
        side_plane11_outer_center = torch.vstack([
            self.outer_tank_center[0],
            self.outer_tank_center[1],
            self.outer_tank_center[2],
        ]
        )   
        side_plane12_outer_center = side_plane11_outer_center - self.outer_tank_thickness * axes11[:,0].unsqueeze(-1)
        side_plane11_inner_center = side_plane12_outer_center - (self.inner_tank_distance[0]) * axes11[:,0].unsqueeze(-1) - (self.outer_tank_size[1] / 2 - (self.outer_tank_thickness + self.inner_tank_size[1]/2 + self.inner_tank_distance[0])) * axes11[:,1].unsqueeze(-1)
        
        #side_plane21_outer_center = self.outer_tank_center.unsqueeze(-1) + self.outer_tank_size[0] / 2 * axes21[:,0].unsqueeze(-1) - self.#outer_tank_size[1] / 2 * axes21[:,1].unsqueeze(-1)        
        #side_plane11_inner_center[1,:] = (side_plane21_outer_center + self.inner_tank_distance[0].unsqueeze(-1) * axes21[:,0].unsqueeze(-1))[1,:]

        side_plane11_inner = RefractingPlane(
            refractive_idx_1=self.refractive_index_water,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes11,
            a=self.inner_tank_size[1],
            b=self.inner_tank_size[2],
            center=side_plane11_inner_center,
        )

        side_plane12_inner_center = side_plane11_inner_center - self.inner_tank_thickness * axes11[:,0].unsqueeze(-1)
        side_plane12_inner = RefractingPlane(
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            axes=axes11, # Same axis as plane11
            a=self.inner_tank_size[1],
            b=self.inner_tank_size[2],
            center=side_plane12_inner_center,
        )

        
        side_plane21_inner_center = side_plane11_inner_center + self.inner_tank_size[0] / 2 * axes21[:,0].unsqueeze(-1) - self.inner_tank_size[1] / 2 * axes21[:,1].unsqueeze(-1)        
        side_plane21_inner = RefractingPlane(
            refractive_idx_1=self.refractive_index_water,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes21,
            a=self.inner_tank_size[1],
            b=self.inner_tank_size[2],
            center=side_plane21_inner_center,
        )


        # Making side_plane_22 parallel to side_plane21 (wall thickness)        
        side_plane22_inner_center = side_plane21_inner_center - self.inner_tank_thickness * axes21[:,0].unsqueeze(-1)
        side_plane22_inner = RefractingPlane(
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            axes=axes21,
            a=self.inner_tank_size[1],
            b=self.inner_tank_size[2],
            center=side_plane22_inner_center,
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
        
        top_plane1_inner_center = torch.vstack([
            side_plane11_inner_center[0],
            side_plane11_inner_center[1],
            side_plane11_inner_center[2],
        ]
        ) - self.inner_tank_size[0] / 2 * axes11[:,0].unsqueeze(-1) + self.inner_tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        top_plane1_inner_center - self.outer_tank_thickness * axes_top[:,0].unsqueeze(-1)

        top_plane2_outer_center = top_plane1_inner_center - self.outer_tank_thickness * axes_top[:,0].unsqueeze(-1)
        small_gap_width = 0.01 #This is the width I'm assuming for the gap between the two tanks. Needs to be refined later
        top_plane1_inner_center = top_plane2_outer_center - small_gap_width * axes_top[:,0].unsqueeze(-1)

        top_plane1_inner = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_water,
            refractive_idx_2=self.refractive_index_acrylic,
            a=self.inner_tank_size[0],
            b=self.inner_tank_size[1],
            center=top_plane1_inner_center
        )

        top_plane2_inner_center = top_plane1_inner_center - self.inner_tank_thickness * axes_top[:,0].unsqueeze(-1)
        top_plane2_inner = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.inner_tank_size[0],
            b=self.inner_tank_size[1],
            center=top_plane2_inner_center
        )
        return side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_inner, top_plane2_inner