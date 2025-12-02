# %% Imports
from ray_tracing_simulator_nnModules_grad import PrismMirror, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
import matplotlib.pyplot as plt
import numpy as np  
import torch
import random
seed = 42
# Python random
random.seed(seed)
# NumPy random
np.random.seed(seed)
# PyTorch random
torch.manual_seed(seed)
import scipy.io as sio
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
torch.autograd.set_detect_anomaly(True)
import datetime
import time
from arenas.prism_arenas import Arena_reprojection_loss_two_cameras_prism_grid_distances
from utils import euclidean_distance
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import pdb
import pytorch3d.transforms as transforms3d
def set_seed(seed=42):
    """Claude's seed function"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed
# set_seed(42)
#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, data, labels_2D, labels_3D, pairwise_distance):
        # Example data
        self.data = data.T
        self.labels_2D = labels_2D.T
        self.labels_3D = labels_3D.T
        self.pairwise_distance = pairwise_distance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels_2D[idx], self.labels_3D[idx], self.pairwise_distance[idx]

#%%
reprojection_errors_iter = torch.zeros(50,1)
closest_distance_loss_iter = torch.zeros(50,1)
pairwise_distance_loss_iter = torch.zeros(50,1)
triangulation_loss_iter = torch.zeros(50,1)

#%%
exp_id = '32'
if 'exp_' in exp_id:
    exp_id = exp_id.removeprefix("exp_")


#%%
load_checkpoint = True
device = torch.device("cpu") # Empirically, CPU seems to work faster for optimization
print(f'Device: {device}')


#%% Continue to load camera calibration results
datatype = torch.float32
pi = torch.tensor(np.pi, dtype=datatype).to(device)
calibration_results_dir = f'/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_{exp_id}/results'
calibration_results_file = 'dotted_grid_pairwise_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
prism_initialization_path = os.path.join(calibration_results_dir, 'prism_initialization.mat')
prism_image_name = 'initialization'
if os.path.isfile(
    os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_0.bmp')
    ):
    prism_image_cam_0_path = os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_0.bmp',
    )
else:
    prism_image_cam_0_path = os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_0.png'
)
if os.path.isfile(
    os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_1.bmp')
    ):
    prism_image_cam_1_path = os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_1.bmp',
    )
else:
    prism_image_cam_1_path = os.path.join(
    os.path.dirname(calibration_results_dir),
    'calibration_grid_images',
    f'{prism_image_name}_cam_1.png',
)
prism_image_cam_0 = plt.imread(prism_image_cam_0_path)    
prism_image_cam_1 = plt.imread(prism_image_cam_1_path)
print('Loaded prism initialization data from {prism_initialization_path}')
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()

model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/exp_32_2025_10_5_22_22_36'
#model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/exp_62_2025_9_4_12_12_48'


def corners_within_limits(cornerA, cornerB, cornerC, cornerD, prism_size):
    if (torch.abs(cornerA[0]) <= prism_size * 0.4 and torch.abs(cornerA[1]) <= prism_size * 0.45 and
        torch.abs(cornerB[0]) <= prism_size * 0.4 and torch.abs(cornerB[1]) <= prism_size * 0.45 and
        torch.abs(cornerC[0]) <= prism_size * 0.4 and torch.abs(cornerC[1]) <= prism_size * 0.45 and
        torch.abs(cornerD[0]) <= prism_size * 0.4 and torch.abs(cornerD[1]) <= prism_size * 0.45):
        return True
    else:
        return False

def get_grid_transformation(points, arena, grid_size, grid_spacing=1., prism_size=20.):
    """
    This method performs two tasks:
    1. Samples a random configuration of the grid over the prism
    2. Returns grid points transformed from the grid's reference frame to the camera's reference frame. The exact transformation is estimated for the top left corner of the grid.
    points: (3, N) tensor of grid points in the grid's reference frame
    theta: rotation in the plane of the prism's top surface
    x,y: translation in the plane of the prism's top surface (0, 0) is the center of the top surface
    prism_size: size of the prism's top square surface (float)
    """
    points1 = points[:3, :]
    points2 = points[3:, :]
    grid_size_horizontal = (grid_size[0] - 1) * grid_spacing
    grid_size_vertical = (grid_size[1] - 1) * grid_spacing
        
    flag = 0
    while flag == 0:
        theta = (torch.rand(1) - 0.5) * 2 * torch.pi / 4
        #theta = torch.tensor(-0.5484)
        x = (torch.rand(1) - 0.5) * prism_size
        y = (torch.rand(1) - 0.5) * prism_size
        #x, y = -2.2512, -6.8978 # Delete this. Only for debugging
        cornerA = torch.tensor([
            x - grid_size_vertical * torch.cos(-pi/4) * torch.sin(theta),
            y + grid_size_vertical * torch.cos(-pi/4) * torch.cos(theta),
            0,
        ])
        cornerB = torch.tensor([
            x + grid_size_horizontal * torch.cos(theta) - grid_size_vertical * torch.cos(-pi/4) * torch.sin(theta),
            y + grid_size_horizontal * torch.sin(theta) + grid_size_vertical * torch.cos(-pi/4) * torch.cos(theta),
            0,
        ])
        cornerC = torch.tensor([
            x + grid_size_horizontal * torch.cos(theta),
            y + grid_size_horizontal * torch.sin(theta),
            grid_size_vertical, # height before rotation
        ])
        # corner D is the top left corner of the grid
        cornerD = torch.tensor([
            x,
            y,
            grid_size_vertical, # height before rotation
        ])
        if corners_within_limits(cornerA, cornerB, cornerC, cornerD, prism_size):            
            grid_corner = torch.tensor([0, 0, grid_size_vertical]).unsqueeze(-1).to(torch.float32)
            flag = 1
        with torch.no_grad():
            _, _, top_plane = arena.prism.get_planes(
            arena.prism.prism_center,
            arena.prism.prism_rotation_6d
        )
        
    scaled_horizontal_axis_prism = pi/4 * torch.tensor([1., 0., 0.], dtype=torch.float32) 
    R_incline = transforms3d.axis_angle_to_matrix(scaled_horizontal_axis_prism)    
    
    scaled_normal_axis_prism = theta * torch.tensor([0., 0., 1.], dtype=torch.float32) 
    R_theta = transforms3d.axis_angle_to_matrix(scaled_normal_axis_prism)    
    
    height_after_inclining = (R_incline @ torch.tensor([0., 0., grid_size_vertical], dtype=torch.float32).unsqueeze(-1))[-1]
    desired_grid_corner_after_inclining = torch.tensor([x, y, height_after_inclining], dtype=torch.float32).unsqueeze(-1)
    points1 = R_theta @ R_incline @ (points1 - grid_corner) + desired_grid_corner_after_inclining
    points2 = R_theta @ R_incline @ (points2 - grid_corner) + desired_grid_corner_after_inclining
    
    # In the grid's reference frame, z should become the negative normal of the top plane, x should become the horizontal axis of the top plane and y should become the negative depth axis of the top plane
    negative_normal = -1 * top_plane.axes[:,0]
    horizontal_axis = top_plane.axes[:,1]
    depth_axis = -1 * top_plane.axes[:,2]
    transformation_matrix = torch.vstack((
        horizontal_axis,
        depth_axis,
        negative_normal,
    )).T
    points1 = transformation_matrix @ (points1) + top_plane.center
    points2 = transformation_matrix @ (points2) + top_plane.center
    points = torch.vstack(
        (points1, points2)
        )
    return points.squeeze(), theta, x, y


def generate_toy_ground_truth(N, grid_spacing, grid_size, arena, noise=0.):
    """
    N: num_pairs
    grid_spacing: spacing between grid points in mm
    grid_size: (num_points_x, num_points_y)
    """
    target_coordinates = torch.zeros((6, N), dtype=torch.float32)
    target_coordinates_grid_ref = torch.zeros_like(target_coordinates)
    pairwise_distance = torch.zeros((N,), dtype=torch.float32)
    grid_x = torch.linspace(0, grid_size[0] - 1, grid_size[0])
    grid_z = torch.linspace(0, grid_size[1] - 1, grid_size[1])
    grid_xi_rand = torch.randint(0, grid_size[0], (N,1))
    grid_zi_rand = torch.randint(0, grid_size[1], (N,1))
    grid_xi_rand_pair = (torch.randint(1, grid_size[0]-1, (N,1)) + grid_xi_rand) % grid_size[0]
    grid_zi_rand_pair = (torch.randint(1, grid_size[1]-1, (N,1)) + grid_zi_rand) % grid_size[1]

    #[grid_x, grid_z] = torch.meshgrid(grid_x, grid_z, indexing='ij')
    #grid_x = grid_x.flatten()
    #grid_z = grid_z.flatten()    
    #grid_xi_rand = torch.arange(N)
    #grid_zi_rand = torch.arange(N)
    for point_id in range(N):
        # Sample a pair of points on the grid in the grid's reference frame
        target_coordinates_grid_ref[:3, point_id] = torch.tensor(
            [grid_x[grid_xi_rand[point_id]] * grid_spacing,
            0.,
            grid_z[grid_zi_rand[point_id]] * grid_spacing,
            ],
            dtype=torch.float32,
        )
        target_coordinates_grid_ref[3:, point_id] = torch.tensor(
            [grid_x[grid_xi_rand_pair[point_id]] * grid_spacing,
            0.,
            grid_z[grid_zi_rand_pair[point_id]] * grid_spacing,
            ],
            dtype=torch.float32,
        )
            
        # Transform the points to the camera's reference frame and place the points appropriately over the prism
        target_coordinates[:, point_id], theta, x, y = get_grid_transformation(
            target_coordinates_grid_ref[:, point_id].unsqueeze(-1),
            arena,
            grid_size,
            grid_spacing,
            prism_size=arena.prism.prism_size[0].item(),
        )

        pairwise_distance[point_id] = euclidean_distance(
            target_coordinates[:3, point_id].unsqueeze(-1),
            target_coordinates[3:, point_id].unsqueeze(-1),
        )
    return target_coordinates, pairwise_distance


# Tensorboard writer
writer = SummaryWriter(log_dir=f'{outputs_dir}/logs/{model_checkpoint_dir}')
print(f'To visualize tensorboard log, run: tensorboard --logdir={outputs_dir}/logs/{model_checkpoint_dir}')

def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr

#%% Freeze parameters
def freeze_camera_parameters(camera):
    for param in camera.parameters():
        param.requires_grad = False

def freeze_individual_planes(prism):
    for param in prism.plane1.parameters():
        param.requires_grad = False
    
    for param in prism.plane2.parameters():
        param.requires_grad = False
    
    for param in prism.plane3.parameters():
        param.requires_grad = False

def freeze_stereocamera(arena):
    arena.focal_length_cam_1.requires_grad = False
    arena.principal_point_pixel_cam_1.requires_grad = False
    arena.stereo_camera_rotation_6d.rotation_6d.requires_grad = False
    arena.T_stereo_cam.requires_grad = False
    arena.stereocam_r1.requires_grad = False

def freeze_prism_parameters_subset(arena):
    arena.prism.prism_size.requires_grad = False
    arena.prism.refractive_index_glass.requires_grad = False

def unfreeze_camera_parameters(camera):
    for param in camera.parameters():
        param.requires_grad = True

def unfreeze_stereocamera(arena):
    arena.focal_length_cam_1.requires_grad = True
    arena.principal_point_pixel_cam_1.requires_grad = True
    arena.stereo_camera_rotation_6d.requires_grad = True
    arena.stereocam_r1.requires_grad = True

def unfreeze_prism_parameters_subset(prism):
    for param in prism.parameters():
        param.requires_grad = True

def unfreeze_all_parameters(arena):
    for param in arena.parameters():
        if not param.requires_grad:
            param.requires_grad = True


#%% Initialize an Arena instance
def initialize_Arena():
    prism_angles = torch.tensor([0., 0., 0.], dtype=torch.float32).to(device)
    prism_center = torch.zeros((3,1), dtype=torch.float32).to(device)
    principal_point_pixel_cam_0 = torch.zeros((2,1), dtype=torch.float32).to(device)
    principal_point_pixel_cam_1 = torch.zeros((2,1), dtype=torch.float32).to(device)
    focal_length_cam_1 = torch.tensor(0.1, dtype=torch.float32).to(device)
    focal_length_cam_2 = torch.tensor(0.1, dtype=torch.float32).to(device)
    R = torch.eye(3, dtype=torch.float32).to(device)
    T = torch.tensor([[60.0, 0.0, 0.0]], dtype=torch.float32).T.to(device)
    radial_dist_coeffs_cam_0 = torch.zeros((3,1), dtype=torch.float32).to(device)
    radial_dist_coeffs_cam_1 = torch.zeros((3,1), dtype=torch.float32).to(device)

    arena = Arena_reprojection_loss_two_cameras_prism_grid_distances(principal_point_pixel_cam_0,
                principal_point_pixel_cam_1, 
                focal_length_cam_1, 
                focal_length_cam_2,
                R,
                T, 
                prism_angles=prism_angles,
                prism_center=prism_center,
                radial_dist_coeffs_cam_0=radial_dist_coeffs_cam_0,
                radial_dist_coeffs_cam_1=radial_dist_coeffs_cam_1,
                prism_size=torch.tensor([80., 20., 20.], dtype=datatype).to(device))
    return arena

arena = initialize_Arena()
freeze_camera_parameters(arena.camera1)
freeze_stereocamera(arena)
freeze_prism_parameters_subset(arena)

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'],
                        strict=False)
with torch.no_grad():
    arena.prism.prism_size.copy_(torch.tensor([20., 20., 20.], dtype=datatype).to(device))
# %% Generate toy dataset
target_coordinates, pairwise_distances = generate_toy_ground_truth(
    32, 1, [9,9], arena
)
assert euclidean_distance(
    target_coordinates[:3,:],
    target_coordinates[3:,:]
).sum() == pairwise_distances.sum()

#%% Plot dataset over prism
target_coordinates_np = target_coordinates.detach().numpy()
fig, ax = arena.prism.visualize_prism()
ax.scatter(
    target_coordinates_np[0,:],
    target_coordinates_np[1,:],
    target_coordinates_np[2,:],
    s=3,
)
ax.scatter(
    target_coordinates_np[3,:],
    target_coordinates_np[4,:],
    target_coordinates_np[5,:],
    s=3,
    c='r'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.close('all')
#%%
def train_two_cams(model, train_loader, criterion, plot=False):    
    model.train()
    virtual_loss = 0.
    real_loss = 0.
    closest_dist_loss = 0.
    total_loss = 0.
    distortion_loss = 0.
    intersection_loss = 0.
    reprojection_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    calibration_skew_loss = 0.
    
    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for i, (input, label_2D, label_3D, pairwise_distance_batch) in enumerate(train_loader):
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            input = input.to(device)
            label_2D = label_2D.to(device)
            label_3D = label_3D.to(device)
            pairwise_distance_batch = pairwise_distance_batch.to(device)
            output = model(
                input.T,
                label_2D.T)
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_pixels_1_from_virtual, recon_pixels_2_from_virtual, recon_pixels_1_to_virtual, recon_pixels_2_to_virtual, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2, pairwise_distance_recon = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_pixels_0_virtual'], output['recon_pixels_1_virtual'], output['recon_pixels_1_to_virtual'], output['recon_pixels_2_to_virtual'], output['distortion_penalty_cam_0'], output['distortion_penalty_cam_1'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance']
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            label_2D_cam_0 = torch.vstack((label_2D[:,:2], label_2D[:,2:4]))
            label_2D_cam_1 = torch.vstack((label_2D[:,4:6], label_2D[:,6:8]))
            virtual_2D_cam_0 = torch.vstack((input[:,:2], input[:,2:4]))
            virtual_2D_cam_1 = torch.vstack((input[:,4:6], input[:,6:8]))
            stacked_label_2D = torch.vstack((label_2D[:,:label_2D.shape[1] // 2], label_2D[:,label_2D.shape[1] // 2:]))
            stacked_label_3D = torch.vstack((label_3D[:,:label_3D.shape[1] // 2], label_3D[:,label_3D.shape[1] // 2:]))
            if plot:                
                rand_ind = torch.randperm(recon_pixels_1.shape[1])
                plt.subplot(121)
                plt.scatter(
                    recon_pixels_1[0,rand_ind].cpu().detach().numpy(),
                    recon_pixels_1[1,rand_ind].cpu().detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    label_2D_cam_0.T[0,rand_ind].cpu().detach().numpy(),
                    label_2D_cam_0.T[1,rand_ind].cpu().detach().numpy(),
                    s=0.5,
                    c='g',
                )
                plt.subplot(122)
                plt.scatter(
                    recon_pixels_2[0,rand_ind].cpu().detach().numpy(),
                    recon_pixels_2[1,rand_ind].cpu().detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    label_2D_cam_1.T[0,rand_ind].cpu().detach().numpy(),
                    label_2D_cam_1.T[1,rand_ind].cpu().detach().numpy(),
                    s=0.5,
                    c='g',
                )


            if recon_pixels_1_to_virtual is not None and recon_pixels_2_to_virtual is not None:
                print('Calculated virtual reprojection error')
                
                overall_reprojection_loss = (euclidean_distance(
                        recon_pixels_1_to_virtual, 
                        virtual_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2_to_virtual, 
                        virtual_2D_cam_1.T).sum()
                    ) / 4
                
                """overall_reprojection_loss = (euclidean_distance(
                        recon_pixels_1_to_virtual, 
                        virtual_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2_to_virtual, 
                        virtual_2D_cam_1.T).sum() + euclidean_distance(
                        recon_pixels_1,
                        label_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2,
                        label_2D_cam_1.T).sum()
                    ) / 8"""
            else:
                if torch.rand(1) < .5:
                    overall_reprojection_loss = (euclidean_distance(
                        recon_pixels_1, 
                        label_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2,
                        label_2D_cam_1.T).sum()) / 4
                else:
                    overall_reprojection_loss = (euclidean_distance(
                        recon_pixels_1_from_virtual, 
                        label_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2_from_virtual,
                        label_2D_cam_1.T).sum()) / 4
            

            triangulation_loss = euclidean_distance(
                                    recon_3D, stacked_label_3D.T
                                ).sum() / 2
            
            distortion_loss = (dist_penalty_1.sum() + dist_penalty_2.sum()) / 2
            intersection_loss = (int_penalty_1.sum() + int_penalty_2.sum()) / 2
            closest_distance_loss = closest_distance.sum()
            tr_loss += triangulation_loss.item()            
            # Recon_real_loss is the pixel error between the reprojected 3D point from real pixels and the real pixel
            loss = (1e1 * overall_reprojection_loss) + (1e10 * intersection_loss) + 0 * distortion_loss + (1e2 * closest_distance_loss) + (1e4 * pairwise_distance_loss) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # virtual_loss += recon_virtual_loss.item()
            # real_loss += recon_real_loss.item()
            closest_dist_loss += closest_distance_loss.item()
            distortion_loss += distortion_loss.item()
            intersection_loss += intersection_loss.item()
            reprojection_loss += overall_reprojection_loss.item()     
            pairwise_dist_loss += pairwise_distance_loss.item()      

        virtual_loss = 0.
        real_loss = 0.

    return total_loss / len(train_loader.dataset), reprojection_loss / len(train_loader.dataset), virtual_loss / len(train_loader.dataset), real_loss / len(train_loader.dataset), closest_dist_loss / len(train_loader.dataset), distortion_loss / len(train_loader.dataset), intersection_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset), pairwise_dist_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion):
    model.eval()
    virtual_loss = 0.
    real_loss = 0.
    closest_dist_loss = 0.
    total_loss = 0.
    distortion_loss = 0.
    total_intersection_loss = 0.
    tr_loss = 0.
    reprojection_loss = 0.
    pairwise_dist_loss = 0.
    with torch.no_grad():
        for i, (input, label_2D, label_3D, pairwise_distance_batch) in enumerate(val_loader):
            # input: virtual pixels from two cameras stacked
            # label_2D: real pixels from two cameras stacked
            # label_3D: target 3D coordinates stacked
            # pairwise_distance_batch: pairwise distances between points in the batch
            input = input.to(device)
            label_2D = label_2D.to(device)
            label_3D = label_3D.to(device)
            pairwise_distance_batch = pairwise_distance_batch.to(device)
            output = model(input.T, label_2D.T)
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_pixels_1_to_virtual, recon_pixels_2_to_virtual, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2, pairwise_distance_recon = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_pixels_1_to_virtual'], output['recon_pixels_2_to_virtual'], output['distortion_penalty_cam_0'], output['distortion_penalty_cam_1'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance']
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error

            """
            recon_pixels = torch.cat((recon_distorted_virtual_pixels_1,
            recon_distorted_virtual_pixels_2), dim=0)
            """

            label_2D_cam_0 = torch.vstack((label_2D[:,:2], label_2D[:,2:4]))
            label_2D_cam_1 = torch.vstack((label_2D[:,4:6], label_2D[:,6:8]))
            virtual_2D_cam_0 = torch.vstack((input[:,:2], input[:,2:4]))
            virtual_2D_cam_1 = torch.vstack((input[:,4:6], input[:,6:8]))
            stacked_label_3D = torch.vstack((label_3D[:,:label_3D.shape[1] // 2], label_3D[:,label_3D.shape[1] // 2:]))

            """
            recon_virtual_loss = (euclidean_distance(
                recon_distorted_virtual_pixels_1,
                label_2D_cam_0.T).sum() + euclidean_distance(
                recon_distorted_virtual_pixels_2, 
                label_2D_cam_1.T).sum()) / 4

            recon_real_loss = (euclidean_distance(
                recon_distorted_real_pixels_1, 
                label_2D_cam_0.T).sum() + euclidean_distance(
                recon_distorted_real_pixels_2, 
                label_2D_cam_1.T).sum()) / 4
            """
            
            if recon_pixels_1_to_virtual is not None and recon_pixels_2_to_virtual is not None:
                print('Calculated virtual reprojection error')
                overall_reprojection_loss = (euclidean_distance(
                        recon_pixels_1_to_virtual, 
                        virtual_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2_to_virtual, 
                        virtual_2D_cam_1.T).sum() + euclidean_distance(
                        recon_pixels_1,
                        label_2D_cam_0.T).sum() + euclidean_distance(
                        recon_pixels_2,
                        label_2D_cam_1.T).sum()
                    ) / 8

            else:
                overall_reprojection_loss = (euclidean_distance(
                recon_pixels_1, 
                label_2D_cam_0.T).sum() + euclidean_distance(
                recon_pixels_2,
                label_2D_cam_1.T).sum()) / 4
                
            triangulation_loss = euclidean_distance(
                                    recon_3D, stacked_label_3D.T
                                ).sum() / 2
            
            distortion_loss = (dist_penalty_1.sum() + dist_penalty_2.sum()) / 2
            intersection_loss = (int_penalty_1.sum() + int_penalty_2.sum()) / 2
            closest_distance_loss = closest_distance.sum()
            loss = (1e1 * overall_reprojection_loss) + (1e10 * intersection_loss) + 0 * distortion_loss + (1e2 * closest_distance_loss) + (1e4 * pairwise_distance_loss)

            total_loss += loss.item()
            #virtual_loss += recon_virtual_loss.item()
            #real_loss += recon_real_loss.item()
            total_intersection_loss += intersection_loss.item()
            distortion_loss += distortion_loss.item()
            closest_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
            reprojection_loss += overall_reprojection_loss.item()
            pairwise_dist_loss += pairwise_distance_loss.item()

        virtual_loss = 0.
        real_loss = 0.

    return total_loss / len(val_loader.dataset), reprojection_loss / len(val_loader.dataset), virtual_loss / len(val_loader.dataset), real_loss / len(val_loader.dataset), closest_dist_loss / len(val_loader.dataset), distortion_loss / len(val_loader.dataset), total_intersection_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset), pairwise_dist_loss / len(val_loader.dataset)


# %% Project to camera
# Process data in chunks of 128 target coordinates to avoid memory issues
real_pixels_cam_0 = torch.zeros((4, target_coordinates.shape[1]), dtype=torch.float32)
real_pixels_cam_1 = torch.zeros_like(real_pixels_cam_0)
virtual_pixels_cam_0 = torch.zeros_like(real_pixels_cam_0)
virtual_pixels_cam_1 = torch.zeros_like(real_pixels_cam_0)
recon_real_loss = torch.zeros((target_coordinates.shape[1],), dtype=torch.float32)
recon_virtual_loss = torch.zeros((target_coordinates.shape[1],), dtype=torch.float32)
chunk_size = 32
arena.virtual_proj_prob_thresh = 1.
with torch.no_grad():
    for i in range(0, target_coordinates.shape[1], chunk_size):
        chunk_tensor = target_coordinates[:, i:i+chunk_size]
        real_pixels_cam_0[:2, i:i+chunk_size], real_pixels_cam_1[:2, i:i+chunk_size], virtual_pixels_cam_0[:2, i:i+chunk_size], virtual_pixels_cam_1[:2, i:i+chunk_size] = arena.project_points_to_image(chunk_tensor[:3,...])
        chunk_tensor = target_coordinates[:, i:i+chunk_size]        
        real_pixels_cam_0[2:, i:i+chunk_size], real_pixels_cam_1[2:, i:i+chunk_size], virtual_pixels_cam_0[2:, i:i+chunk_size], virtual_pixels_cam_1[2:, i:i+chunk_size] = arena.project_points_to_image(chunk_tensor[3:,...])

        output = arena(
        torch.vstack((virtual_pixels_cam_0[:, i:i+chunk_size].to(device),
                    virtual_pixels_cam_1[:, i:i+chunk_size].to(device))),
        torch.vstack((real_pixels_cam_0[:, i:i+chunk_size].to(device),
        real_pixels_cam_1[:, i:i+chunk_size].to(device))),        
        )
        recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, int_penalty_1, int_penalty_2, pairwise_distance_, recon_pixels_1_to_virtual, recon_pixels_2_to_virtual = output['recon_3D'],output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance'], output['recon_pixels_1_to_virtual'], output['recon_pixels_2_to_virtual']

        recon_real_loss[i:i+chunk_size] = (euclidean_distance(
                recon_pixels_1[:, :chunk_size],
                real_pixels_cam_0[:2, i:i+chunk_size]) + euclidean_distance(
                recon_pixels_2[:, :chunk_size],
                real_pixels_cam_1[:2, i:i+chunk_size])) / 2
        reprojection_loss_1_to_virtual = euclidean_distance(
        recon_pixels_1_to_virtual[:, :chunk_size], virtual_pixels_cam_0[:2, i:i+chunk_size]
        )
        reprojection_loss_2_to_virtual = euclidean_distance(
            recon_pixels_2_to_virtual[:, :chunk_size], virtual_pixels_cam_1[:2, i:i+chunk_size]
        )
        recon_virtual_loss[i:i+chunk_size] = (reprojection_loss_1_to_virtual + reprojection_loss_2_to_virtual) / 2    


# %%
class CalibrationDataset(Dataset):
    def __init__(self, data, labels_2D, labels_3D, pairwise_distance):
        # Example data
        self.data = data.T
        self.labels_2D = labels_2D.T
        self.labels_3D = labels_3D.T
        self.pairwise_distance = pairwise_distance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels_2D[idx], self.labels_3D[idx], self.pairwise_distance[idx]


pixels_virtual_two_cams = torch.vstack(
    (virtual_pixels_cam_0, virtual_pixels_cam_1)
)
pixels_real_two_cams = torch.vstack(
    (real_pixels_cam_0, real_pixels_cam_1)
)

#%%
arena.visualize(pixels_virtual_two_cams.to(device), color_labels=True)
#plt.savefig(f'{outputs_dir}/initialized_arena.png')
output = arena(
    pixels_virtual_two_cams.to(device),
    pixels_real_two_cams.to(device)
)
recon_3D_init, closest_distance_init, recon_pixels_1_init, recon_pixels_2_init, pairwise_distance_init = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['pairwise_distance']

pairwise_distance_loss_init = torch.abs(pairwise_distance_init - pairwise_distances).mean()
pixels_real_cam_0_test_stacked = torch.hstack((pixels_real_two_cams[:2,:], pixels_real_two_cams[2:4,:]))
pixels_real_cam_1_test_stacked = torch.hstack((pixels_real_two_cams[4:6,:], pixels_real_two_cams[6:8,:]))
recon_real_loss_init = (euclidean_distance(
                recon_pixels_1_init, 
                pixels_real_cam_0_test_stacked).mean() + euclidean_distance(
                recon_pixels_2_init, 
                pixels_real_cam_1_test_stacked).mean()) / 2
target_coordinates_stacked = torch.hstack((target_coordinates[:3,:], target_coordinates[3:,:]))
triangulation_loss_init = euclidean_distance(
                        recon_3D_init, target_coordinates_stacked
                    ).mean()    
print(f'Initial real pixel reprojection error: {recon_real_loss_init}, initial closest distance: {closest_distance_init.mean()}, initial pairwise distance error: {pairwise_distance_loss_init}, initial triangulation error: {triangulation_loss_init}')
"""reprojection_errors_iter[iter] = recon_real_loss_init.item()
closest_distance_loss_iter[iter] = closest_distance_init.mean().item()
pairwise_distance_loss_iter[iter] = pairwise_distance_loss_init.item()
triangulation_loss_iter[iter] = triangulation_loss_init.item()"""
#pdb.set_trace()
#%%

dataset = CalibrationDataset(pixels_virtual_two_cams, pixels_real_two_cams, target_coordinates, pairwise_distances)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
print(f'Splitting grid point pairs into {train_size} : {val_size} ratio')
pixels_virtual_two_cams_train, pixels_virtual_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )

batch_size = 128
train_loader = DataLoader(pixels_virtual_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_virtual_two_cams_val, batch_size=batch_size, shuffle=False)

# %% Optimize
optimizer = optim.Adam(arena.parameters(), lr=1e-3
                       )
criterion = torch.nn.MSELoss()
with torch.no_grad():
    arena.prism.prism_size.copy_(torch.tensor([80., 20., 20.], dtype=datatype).to(device))
    prism_center = arena.prism.prism_center.clone()
    prism_center[0] -= 1.
    arena.prism.prism_center.copy_(prism_center).to(device)
arena.to(device)
pixels_virtual_two_cams = pixels_virtual_two_cams.to(device)

# %% Training loop
train_loss_array = []
train_virtual_loss_array = []
train_real_loss_array = []
train_closest_distance_loss_array = []
train_distortion_loss_array = []
train_intersection_loss_array = []

val_loss_array = []
val_virtual_loss_array = []
val_real_loss_array = []
val_closest_distance_loss_array = []
val_distortion_loss_array = []
val_intersection_loss_array = []

gt_train_loss_array = []
closest_distance_train_loss_array = []
gt_val_loss_array = []
closest_distance_val_loss_array = []
best_loss = 1e100

plot = False
arena.virtual_proj_prob_thresh = 0. # probability of calculating virtual reprojection error and using it for backprop
num_epochs = 200

for epoch in tqdm(range(0, num_epochs)):
    if epoch == 75:
        change_lr(optimizer, lr=5e-3)

    if epoch == 150:
        change_lr(optimizer, lr=1e-4)

    if epoch == 200:
        change_lr(optimizer, lr=5e-5)

    if epoch == 250:
        change_lr(optimizer, lr=1e-5)

    if epoch == 300:
        change_lr(optimizer, lr=1e-6)
        unfreeze_all_parameters(arena)
    
    if epoch == 800:
        change_lr(optimizer, lr=5e-5)

    train_loss, train_reprojection_loss, train_virtual_loss, train_real_loss, train_closest_dist_loss, train_distortion_loss, train_intersection_loss, triangulation_loss, train_pairwise_distance_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    
    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train_loss: {train_loss}, reprojection loss: {train_reprojection_loss}, closest_dist_loss: {train_closest_dist_loss}, triangulation loss : {triangulation_loss}, intersection loss: {train_intersection_loss}, distortion loss: {train_distortion_loss}')
        if epoch % 200 == 0:
            plot = True
    train_loss_array.append(train_loss)
    train_virtual_loss_array.append(train_virtual_loss)
    train_real_loss_array.append(train_real_loss)
    train_closest_distance_loss_array.append(train_closest_dist_loss)
    train_distortion_loss_array.append(train_distortion_loss)
    train_intersection_loss_array.append(train_intersection_loss)


    val_loss, val_reprojection_loss, val_virtual_loss, val_real_loss, val_closest_dist_loss, val_distortion_loss, val_intersection_loss, triangulation_loss, val_pairwise_distance_loss = validate(model=arena,
             val_loader=val_loader,
             criterion=criterion,
             )

    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: val_loss: {val_loss}, reprojection error: {val_reprojection_loss}, closest_dist_loss: {val_closest_dist_loss}, triangulation loss {triangulation_loss}, distortion loss: {val_distortion_loss}, intersection loss: {val_intersection_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'val_loss': val_loss,
                    'reprojection_loss': val_reprojection_loss,
                    'closest_dist_loss': val_closest_dist_loss,
                    'distortion_loss': val_distortion_loss,
                    'intersection_loss': val_intersection_loss,
                    'train_pairwise_distance_loss': train_pairwise_distance_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}_toy_model.pth')
                
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint_toy_model.pth'),
        torch.save(
            arena.state_dict(),
            f'{model_checkpoint_dir}/best_model_weights_only_toy_model.pth',
            _use_new_zipfile_serialization=False,
        )
        print(f'Found better model with validation loss for epoch {epoch}: val_loss: {val_loss}, reprojection error: {val_reprojection_loss}, closest_dist_loss: {val_closest_dist_loss}, triangulation loss {triangulation_loss}, pairwise_distance_loss: {val_pairwise_distance_loss}, intersection loss: {val_intersection_loss}')
    gt_train_loss_array.append(train_loss)
    gt_val_loss_array.append(val_loss)
    closest_distance_train_loss_array.append(train_closest_dist_loss)
    closest_distance_val_loss_array.append(val_closest_dist_loss)

# %%
