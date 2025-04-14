# %% Imports
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
import matplotlib.pyplot as plt
import numpy as np  
import torch
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


#%% Load camera calibration results
load_checkpoint = False
parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, help="Experiment ID")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = torch.tensor(np.pi, dtype=torch.float64).to(device)

exp_id = args.exp_id
calibration_results_dir = f'/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_{exp_id}/results/'
calibration_results_file = 'dotted_grid_pairwise_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
prism_initialization_path = os.path.join(calibration_results_dir, 'prism_initialization.mat')
print('Loaded prism initialization data from {prism_initialization_path}')
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/exp_{exp_id}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_pairwise'], dtype=torch.float64).T - 1.
virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_pairwise'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_pairwise'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_pairwise'], dtype=torch.float64).T - 1.
target_coordinates = torch.tensor(mat['worldPoints_pairwise'], dtype=torch.float64).T
pairwise_distance = torch.tensor(mat['pairwise_distances'][:,0]).to(torch.float64)
stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64).to(device)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64).to(device)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64).to(device)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T.to(device) + 0.

#principal_point_pixel_cam_0 = torch.tensor([640., 512.]).to(torch.float64) 
#principal_point_pixel_cam_1 = torch.tensor([640., 512.]).to(torch.float64) 

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64).to(device)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64).to(device)

#focal_length_cam_1 = 5206. 
#focal_length_cam_2 = 5206. 

focal_length_cam_1 = (K1[0,0] + K1[1,1]) /  2
focal_length_cam_2 = (K2[0,0] + K2[1,1]) /  2

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
    arena.stereo_camera_angles.requires_grad = False
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
    arena.stereo_camera_angles.requires_grad = True
    arena.stereocam_r1.requires_grad = True

def unfreeze_prism_parameters_subset(prism):
    for param in prism.parameters():
        param.requires_grad = True

def unfreeze_all_parameters(arena):
    for param in arena.parameters():
        if not param.requires_grad:
            param.requires_grad = True


#%% Prism corners third plane 
prism_annotated_face = 'first'
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/prism_corners_third_plane.mat'
#prism_corners_path = f'{calibration_results_dir}/prism_corners_{prism_annotated_face}_plane.mat'
prism_corners = sio.loadmat(prism_corners_path)['worldPoints']
prism3_axes = torch.zeros(3,3).to(torch.float64).to(device)
prism3_axes[:,1] = torch.stack(
    (torch.tensor(prism_corners[1,:] - prism_corners[0,:], dtype=torch.float64).to(device),
    )
).mean(dim=0)
prism3_axes[:,2] = torch.stack(
    (torch.tensor(prism_corners[3,:] - prism_corners[0,:], dtype=torch.float64).to(device),
     torch.tensor(prism_corners[2,:] - prism_corners[1,:], dtype=torch.float64).to(device)
     )
).mean(dim=0)

if prism_annotated_face == 'third':
    prism3_axes[:,0] = torch.linalg.cross(prism3_axes[:,1], prism3_axes[:,2])
    prism3_axes = prism3_axes / torch.linalg.norm(prism3_axes, dim=0)
    prism1_axes = torch.mm(rotx(pi/2), prism3_axes)

    prism_a = 20.
    prism_b = 20.
    prism3_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0).to(device)
    prism1_center = prism3_center + prism_b/2 * prism1_axes[:,0] - prism_b/2 * prism1_axes[:,2] 

elif prism_annotated_face == 'first':
    prism1_axes = torch.zeros(3,3).to(torch.float64)
    prism1_axes[:,1] = torch.stack(
    (torch.tensor(prism_corners[1,:] - prism_corners[0,:], dtype=torch.float64).to(device),
    )
    ).mean(dim=0)
    prism1_axes[:,2] = torch.stack(
    (torch.tensor(prism_corners[3,:] - prism_corners[0,:], dtype=torch.float64).to(device),
     torch.tensor(prism_corners[2,:] - prism_corners[1,:], dtype=torch.float64).to(device)
     )
    ).mean(dim=0)
    prism1_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0).to(device)
    prism1_axes[:,0] = torch.linalg.cross(prism1_axes[:,1], prism1_axes[:,2])
    prism1_axes = prism1_axes / torch.linalg.norm(prism1_axes, dim=0)

plane = Plane(axes=prism1_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64).to(device)
prism_angles = torch.tensor([-1.8710,  1.4078, -1.9315], dtype=torch.float64).to(device) # Some old optimization results
prism1_center[2] = 250.

#%% Initialize an Arena instance
#prism_angles = torch.tensor([-1.5341, 1.3650, -1.5638], dtype=torch.float64)
#prism_center = torch.tensor([-6.5058, -5.6897, 149.0993], dtype=torch.float64)
prism_initializations = sio.loadmat(prism_initialization_path)
prism_center = torch.tensor(prism_initializations['location_prism']).to(torch.float64).T.to(device)
prism_axes = torch.tensor(prism_initializations['axes_prism']).to(torch.float64).to(device)
plane = Plane(axes=prism_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64).to(device)

arena = Arena_reprojection_loss_two_cameras_prism_grid_distances(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            prism_angles=prism_angles,
            prism_center=prism_center)
pixels_virtual_two_cams = torch.vstack((virtual_pixels_cam_0, virtual_pixels_cam_1))
pixels_real_two_cams = torch.vstack((undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1))
freeze_camera_parameters(arena.camera1)
freeze_stereocamera(arena)
freeze_prism_parameters_subset(arena)

#%% Training and validation functions
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
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2, pairwise_distance_recon = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['distortion_penalty_cam_0'], output['distortion_penalty_cam_1'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance']
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            label_2D_cam_0 = torch.vstack((label_2D[:,:2], label_2D[:,2:4]))
            label_2D_cam_1 = torch.vstack((label_2D[:,4:6], label_2D[:,6:8]))
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
                    label_2D_cam_1.T[1,rand_ind].cpu().detach().numpy(),
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

            # Ground truth pixel error calculated using triangulation of refracted pixels
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
            tr_loss += triangulation_loss.item()            
            # Recon_real_loss is the pixel error between the reprojected 3D point from real pixels and the real pixel
            loss = (1e1 * overall_reprojection_loss) + (1e3 * intersection_loss) + 0 * distortion_loss + (1e1 * closest_distance_loss) + (5e3 * pairwise_distance_loss)
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
    intersection_loss = 0.
    tr_loss = 0.
    reprojection_loss = 0.
    pairwise_dist_loss = 0.
    with torch.no_grad():
        for i, (input, label_2D, label_3D, pairwise_distance_batch) in enumerate(val_loader):
            input = input.to(device)
            label_2D = label_2D.to(device)
            label_3D = label_3D.to(device)
            pairwise_distance_batch = pairwise_distance_batch.to(device)
            output = model(input.T, label_2D.T)
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2, pairwise_distance_recon = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['distortion_penalty_cam_0'], output['distortion_penalty_cam_1'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance']
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error

            """
            recon_pixels = torch.cat((recon_distorted_virtual_pixels_1,
            recon_distorted_virtual_pixels_2), dim=0)
            """

            label_2D_cam_0 = torch.vstack((label_2D[:,:2], label_2D[:,2:4]))
            label_2D_cam_1 = torch.vstack((label_2D[:,4:6], label_2D[:,6:8]))
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
            loss = (1e1 * overall_reprojection_loss) + (1e3 * intersection_loss) + (0 * distortion_loss) + (1e1 * closest_distance_loss) + (5e3 * pairwise_distance_loss)

            total_loss += loss.item()
            #virtual_loss += recon_virtual_loss.item()
            #real_loss += recon_real_loss.item()
            intersection_loss += intersection_loss.item()
            distortion_loss += distortion_loss.item()
            closest_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
            reprojection_loss += overall_reprojection_loss.item()
            pairwise_dist_loss += pairwise_distance_loss.item()

        virtual_loss = 0.
        real_loss = 0.

    return total_loss / len(val_loader.dataset), reprojection_loss / len(val_loader.dataset), virtual_loss / len(val_loader.dataset), real_loss / len(val_loader.dataset), closest_dist_loss / len(val_loader.dataset), distortion_loss / len(val_loader.dataset), intersection_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset), pairwise_dist_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_virtual_two_cams.to(device), color_labels=True)
#plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1024
pixels_virtual_two_cams = pixels_virtual_two_cams.to(device)
pixels_real_two_cams = pixels_real_two_cams.to(device)
target_coordinates = target_coordinates.to(device)
pairwise_distance = pairwise_distance.to(device)
rand_ind = torch.randperm(pixels_virtual_two_cams.shape[1]).to(device)
test_dataset_size = 150
pixels_virtual_two_cams_test = pixels_virtual_two_cams[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]
pixels_real_two_cams_test = pixels_real_two_cams[:, rand_ind[:test_dataset_size]]
pairwise_distance_test = pairwise_distance[rand_ind[:test_dataset_size]]

pixels_virtual_two_cams = pixels_virtual_two_cams[:,rand_ind[test_dataset_size:]]
pixels_real_two_cams = pixels_real_two_cams[:, rand_ind[test_dataset_size:]]
target_coordinates = target_coordinates[:, rand_ind[test_dataset_size:]]
pairwise_distance_train_val = pairwise_distance[rand_ind[test_dataset_size:]]

dataset = CalibrationDataset(pixels_virtual_two_cams, pixels_real_two_cams, target_coordinates, pairwise_distance_train_val)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_virtual_two_cams_train, pixels_virtual_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_virtual_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_virtual_two_cams_val, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(arena.parameters(), lr=1e-2
                       )
criterion = torch.nn.MSELoss()
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
#%%
num_epochs = 1000

plot = False
for epoch in tqdm(range(0, num_epochs)):
    if epoch == 75:
        change_lr(optimizer, lr=1e-2)

    if epoch == 150:
        change_lr(optimizer, lr=5e-3)

    if epoch == 200:
        change_lr(optimizer, lr=1e-3)

    if epoch == 250:
        change_lr(optimizer, lr=5e-4)

    if epoch == 300:
        change_lr(optimizer, lr=1e-4)
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
    
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/val_reprojection_error', val_reprojection_loss, epoch)
    writer.add_scalar('Loss/val_closest_distance_error', val_distortion_loss, epoch)
    writer.add_scalar('Loss/val_triangulation_error', triangulation_loss, epoch)
    writer.add_scalar('Loss/val_pairwise_distance_error', val_pairwise_distance_loss, epoch)
    writer.add_scalar('Loss/val_intersection_error', val_intersection_loss, epoch)
    writer.add_scalar('Loss/val_distortion_error', val_distortion_loss, epoch)
    writer.add_scalar('Parameter/prism/refractive_index_glass', arena.prism.refractive_index_glass, epoch)

    writer.add_scalars('Parameter/prism_angles', {
        'Angle0':arena.prism.prism_angles[0],
         'Angle1':arena.prism.prism_angles[1],
          'Angle2':arena.prism.prism_angles[2]},
            epoch)
    writer.add_scalars('Parameter/prism_size', {
        'Size0':arena.prism.prism_size[0],
         'Size1':arena.prism.prism_size[1],
          'Size2':arena.prism.prism_size[2]},
            epoch)
    writer.add_scalars('Parameter/prism_center', {
        'Center0':arena.prism.prism_center[0],
         'Center1':arena.prism.prism_center[1],
          'Center2':arena.prism.prism_center[2]},
            epoch)
    writer.add_scalar('Parameter/focal_length_pixels_0', arena.camera1.focal_length_pixels, epoch)
    writer.add_scalars('Parameter/camera1_principal_point', {
        'Angle0':arena.camera1.principal_point_pixel[0],
         'Angle1':arena.camera1.principal_point_pixel[1]},
            epoch)

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
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
                
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth'),
        torch.save(
            arena.state_dict(),
            f'{model_checkpoint_dir}/best_model_weights_only.pth',
            _use_new_zipfile_serialization=False,
        )
        print(f'Found better model with validation loss for epoch {epoch}: val_loss: {val_loss}, reprojection error: {val_reprojection_loss}, closest_dist_loss: {val_closest_dist_loss}, triangulation loss {triangulation_loss}, pairwise_distance_loss: {val_pairwise_distance_loss}, intersection loss: {val_intersection_loss}')
    gt_train_loss_array.append(train_loss)
    gt_val_loss_array.append(val_loss)
    closest_distance_train_loss_array.append(train_closest_dist_loss)
    closest_distance_val_loss_array.append(val_closest_dist_loss)

# %% Validate the model
PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

output = arena(pixels_virtual_two_cams_test, pixels_real_two_cams_test)
recon_3D_test, closest_dist_test, recon_pixels_1, recon_pixels_2, recon_3D_real, real_3D_virtual, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2, pairwise_distance_test_ = output['recon_3D'], output['closest_distance'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_3D_real'], output['recon_3D_virtual'], output['distortion_penalty_cam_0'], output['distortion_penalty_cam_1'], output['intersection_penalty_1'], output['intersection_penalty_2'], output['pairwise_distance']
pairwise_distance_loss = torch.abs(pairwise_distance_test_ - pairwise_distance_test).mean()
target_coordinates_test_stacked = torch.hstack((target_coordinates_test[:3,:], target_coordinates_test[3:,:]))
triangulation_loss = euclidean_distance(
    recon_3D_test, target_coordinates_test_stacked
).mean()

pixels_real_cam_0_test_stacked = torch.hstack((pixels_real_two_cams_test[:2,:], pixels_real_two_cams_test[2:4,:]))
pixels_real_cam_1_test_stacked = torch.hstack((pixels_real_two_cams_test[4:6,:], pixels_real_two_cams_test[6:8,:]))
recon_real_loss = (euclidean_distance(
                recon_pixels_1, 
                pixels_real_cam_0_test_stacked).mean() + euclidean_distance(
                recon_pixels_2, 
                pixels_real_cam_1_test_stacked).mean()) / 2

print(f'Triangulation loss: {triangulation_loss}')
print(f'Distortion penalty: {(dist_penalty_1 + dist_penalty_2).mean()}')
print(f'Real pixel loss: {recon_real_loss.mean()}')
print(f'Pairwise distance loss: {pairwise_distance_loss.mean()}')

reprojection_loss_1 = euclidean_distance(
    recon_pixels_1, pixels_real_cam_0_test_stacked
)
reprojection_loss_2 = euclidean_distance(
    recon_pixels_2, pixels_real_cam_1_test_stacked
)

print(f'Reprojection error for two cameras: {reprojection_loss_1.mean()}, {reprojection_loss_2.mean()}')

plt.figure(figsize=(15,15))
plt.scatter(
    recon_pixels_1[0,:].detach().numpy(),
    recon_pixels_1[1,:].detach().numpy(),
    color='g',
    marker='o',
    label='Estimate',
)
plt.scatter(
    pixels_real_cam_0_test_stacked[0,:].detach().numpy(),
    pixels_real_cam_0_test_stacked[1,:].detach().numpy(),
    color='r',
    marker='x',
    label='Ground truth',
)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('X (pixels)', fontsize=22)
ax.set_ylabel('Y (pixels)', fontsize=22)
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
ax.set_title(f'Reprojection error: {reprojection_loss_1.mean():.2f} pixels', fontsize=16)
plt.legend(fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/reprojection_loss.png')


#%%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    target_coordinates_test_stacked[0,:].detach().numpy(),
    target_coordinates_test_stacked[1,:].detach().numpy(),
    target_coordinates_test_stacked[2,:].detach().numpy(),
    color='r',
    s=5,
    label='Ground truth',
)
ax.scatter(
    recon_3D_test[0,:].detach().numpy(),
    recon_3D_test[1,:].detach().numpy(),
    recon_3D_test[2,:].detach().numpy(),
    color='g',
    s=5,
    label='Estimate',
)
ax.set_xlabel('X (mm)', fontsize=22)
ax.set_ylabel('Y (mm)', fontsize=22)
ax.set_zlabel('Z (mm)', fontsize=22)
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
ax.set_zticklabels(ax.get_zticks(), fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/3D_scatter.png')


#%% Test loss
#%% Make plots after training
plt.figure(figsize=(15,15))
epochs = np.arange(0,len(gt_train_loss_array))
plt.plot(epochs,gt_train_loss_array, color='b', 
         label='Ground truth train loss')
plt.plot(epochs,closest_distance_train_loss_array, 
         color='r', 
         label='Closest distance train loss')
plt.plot(epochs, gt_val_loss_array, color='b', 
         label='Ground truth val loss',
         linestyle='--')
plt.plot(epochs, closest_distance_val_loss_array, 
         color='r', 
         label='Closest distance val loss',
         linestyle='--')
plt.legend(fontsize=18)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss (mm)', fontsize=22)
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
ax.set_zticklabels(ax.get_zticks(), fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/training_loss.png')


#%% Visualize trained arena
arena.visualize(pixels_virtual_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
