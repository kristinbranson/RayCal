# %% Imports
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, visualize_camera_configuration, closest_point, rotx, get_rot_mat, roty, rotz
import matplotlib.pyplot as plt
import numpy as np  
import torch
import scipy.io as sio
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
#from torchsummary import summary
from torch.utils.data import DataLoader, random_split, Dataset
pi = torch.tensor(np.pi, dtype=torch.float64)
torch.autograd.set_detect_anomaly(True)
import datetime
import time
from arenasEfficient import Arena_single_camera_prism_grid_image
from utils import euclidean_distance
from torch.utils.tensorboard import SummaryWriter

#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, labels_virtual_2D, labels_real_2D, labels_3D, grid_):
        # Example data
        self.labels_virtual_2D = labels_virtual_2D.T
        self.labels_real_2D = labels_real_2D.T
        self.labels_3D = labels_3D.T
        self.grid = grid_.T

    def __len__(self):
        return len(self.labels_virtual_2D)

    def __getitem__(self, idx):
        return self.labels_virtual_2D[idx], self.labels_real_2D[idx], self.labels_3D[idx], self.grid[idx]


#%% Load camera calibration results
load_checkpoint = False
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_3/results/'
calibration_results_file = 'dotted_grid_grid_image_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
virtual_pixels_cam_0 = torch.tensor(mat['imagePoints_av_mat'], dtype=torch.float64).T - 1.
virtual_pixels_cam_1 = torch.tensor(mat['imagePoints_bv_mat'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_0 = torch.tensor(mat['imagePoints_a_mat'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_1 = torch.tensor(mat['imagePoints_b_mat'], dtype=torch.float64).T - 1.
target_coordinates = torch.tensor(mat['worldPoints_mat'], dtype=torch.float64).T
grid = torch.tensor(mat['grid'], dtype=torch.float64).T.unsqueeze(-1)
stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T + 0.

principal_point_pixel_cam_0 = torch.tensor([640., 512.]).to(torch.float64)
principal_point_pixel_cam_1 = torch.tensor([640., 512.]).to(torch.float64)

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

#focal_length_cam_1 = 5208. 
#focal_length_cam_2 = 5208. 

focal_length_cam_1 = (K1[0,0] + K1[1,1]) /  2
focal_length_cam_2 = (K2[0,0] + K2[1,1]) /  2

# Tensorboard writer
writer = SummaryWriter(log_dir=f'{outputs_dir}/logs/{model_checkpoint_dir}')
print(f'To visualize tensorboard log, run: tensorboard --logdir={outputs_dir}/logs/{model_checkpoint_dir}')


#%% Freeze parameters
def freeze_camera_parameters(camera, distortion=True):
    for param in camera.parameters():
        param.requires_grad = False
    if distortion:
        camera.r1.requires_grad = True

def unfreeze_camera_parameters(camera):
    for param in camera.parameters():
        param.requires_grad = True

def freeze_individual_planes(prism):
    for param in prism.plane1.parameters():
        param.requires_grad = False
    
    for param in prism.plane2.parameters():
        param.requires_grad = False
    
    for param in prism.plane3.parameters():
        param.requires_grad = False    

def freeze_stereocamera(arena, distortion=True):
    arena.focal_length_cam_1.requires_grad = False
    arena.principal_point_pixel_cam_1.requires_grad = False
    arena.stereo_camera_angles.requires_grad = False
    if not distortion:
        arena.stereocam_r1.requires_grad = False
    else:
        arena.stereocam_r1.requires_grad = True
    

#%% Prism corners third plane 
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/prism_corners_third_plane.mat'
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_2/results/prism_corners_first_plane.mat'
#prism_corners_path = f'{calibration_results_dir}/prism_corners_first_plane.mat'
prism_corners = sio.loadmat(prism_corners_path)['worldPoints']
prism3_axes = torch.zeros(3,3).to(torch.float64)
prism3_axes[:,1] = torch.stack(
    (torch.tensor(prism_corners[1,:] - prism_corners[0,:], dtype=torch.float64),
    )
).mean(dim=0)
prism3_axes[:,2] = torch.stack(
    (torch.tensor(prism_corners[3,:] - prism_corners[0,:], dtype=torch.float64),
     torch.tensor(prism_corners[2,:] - prism_corners[1,:], dtype=torch.float64)
     )
).mean(dim=0)
prism3_axes[:,0] = torch.linalg.cross(prism3_axes[:,1], prism3_axes[:,2])
prism3_axes = prism3_axes / torch.linalg.norm(prism3_axes, dim=0)
prism1_axes = torch.mm(rotx(pi/2), prism3_axes)

prism_a = 20.
prism_b = 20.
prism3_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0)
prism1_center = prism3_center + prism_b/2 * prism1_axes[:,0] - prism_b/2 * prism1_axes[:,2] 
plane = Plane(axes=prism1_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64)
max_norm = 10.

#%% Initialize an Arena instance
prism_angles = torch.tensor([-1.8710,  1.4078, -1.9315], dtype=torch.float64)
prism1_center = torch.tensor([ -5.1633,  -9.7246, 145.1504], dtype=torch.float64)
prism_distance = torch.tensor(130.) # Not used if you're using fiduciary markers for initialization
arena = Arena_single_camera_prism_grid_image(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            prism_angles=prism_angles,
            prism_center=prism1_center)
pixels_virtual_two_cams = torch.vstack((virtual_pixels_cam_0, virtual_pixels_cam_1))
pixels_real_two_cams = torch.vstack((undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1))
#freeze_camera_parameters(arena.camera1)


#%% Training and validation functions
def train_two_cams(model, train_loader, criterion, plot=False):    
    model.train()
    repr_loss = 0.
    dist_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.
    distort_loss = 0.
    grid_norm_loss = 0.

    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for i, (input, label_2D, label_3D, grid_3D) in enumerate(train_loader):
            label_2D = label_2D.T.flatten(1)
            input = input.T.flatten(1)   
            label_3D = label_3D.T
            optimizer.zero_grad()
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_real_loss, intersection_penalty, distortion_penalty = model(label_2D, input)
            triangulation_loss = euclidean_distance(
                            label_3D.flatten(1), recon_3D
                            ).sum() / grid_3D.shape[1] # Averaged over all gridpoints
            recon_3D = recon_3D.reshape_as(label_3D)
            
            if epoch == 0 and i == 0:
                print(f'Loss for first iteration: Closest_distance_loss: {closest_distance.mean()}, reprojection loss: {recon_real_loss.mean()}')
            intersection_loss = intersection_penalty.sum()
            distortion_loss = distortion_penalty.sum()
            
            cam_0_pixel_gt = torch.hstack((label_2D[:2,:], label_2D[2:4,:]))
            if plot:                
                rand_ind = torch.randperm(label_2D.shape[-1])
                plt.subplot(121)
                plt.scatter(
                    recon_distorted_pixels_1[0,rand_ind].detach().numpy(),
                    recon_distorted_pixels_1[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    cam_0_pixel_gt[0,rand_ind].detach().numpy(),
                    cam_0_pixel_gt[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='g',
                )  
            recon_3D = recon_3D - recon_3D[:, 0, :].unsqueeze(1)
            d_grid = torch.abs(torch.norm(recon_3D.flatten(1), 
                                p=2,
                                dim=0) - torch.norm(grid_3D.T.flatten(1),
                                                    p=2,
                                                    dim=0)
            ).sum() / grid_3D.shape[1]
            reprojection_loss = recon_real_loss.sum() / grid_3D.shape[1]
            closest_distance_loss = closest_distance.sum() / grid_3D.shape[1]
            loss = 1e1 * reprojection_loss.sum()  + 0 * triangulation_loss + 1e1 * closest_distance_loss + 1e3 * intersection_loss + 0 * distortion_loss + 5e3 * d_grid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arena.parameters(), max_norm=max_norm)
            optimizer.step()
            repr_loss += reprojection_loss.item()
            tr_loss += triangulation_loss.item()
            dist_loss += closest_distance_loss.item()   
            distort_loss += distortion_loss.item()    
            grid_norm_loss += d_grid.item()   
            total_loss += loss.item()
    return total_loss / len(train_loader.dataset), repr_loss / len(train_loader.dataset), dist_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset), intersection_loss / len(train_loader.dataset), distort_loss / len(train_loader.dataset), grid_norm_loss / len(train_loader.dataset)


def validate(model, val_dataloader, criterion):
    model.eval()
    val_repr_loss = 0.
    val_dist_loss = 0.
    tr_loss = 0.
    grid_norm_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.
    distort_loss = 0.
    with torch.no_grad():
        for input, label_2D, label_3D, grid_3D in val_dataloader:
            label_2D = label_2D.T.flatten(1)
            input = input.T.flatten(1)
            label_3D = label_3D.T
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_real_loss, intersection_penalty, distortion_penalty = model(label_2D, input)
            recon_3D = recon_3D.reshape_as(label_3D)

            reprojection_loss = recon_real_loss.sum() / grid_3D.shape[1]
            closest_distance_loss = closest_distance.sum() / grid_3D.shape[1]
            intersection_loss = intersection_penalty.sum()
            triangulation_loss = euclidean_distance(
                            label_3D.flatten(1), recon_3D.flatten(1)
                            ).sum() / grid_3D.shape[1]
            recon_3D = recon_3D - recon_3D[:, 0, :].unsqueeze(1)
            d_grid = torch.abs(torch.norm(recon_3D.flatten(1), 
                                p=2,
                                dim=0) - torch.norm(grid_3D.T.flatten(1),
                                                    p=2,
                                                    dim=0) 
            ).sum() / grid_3D.shape[1]
            distortion_loss = distortion_penalty.sum()
            loss = 1e1 * reprojection_loss.sum()  + 0 * triangulation_loss + 1 * closest_distance_loss + 1e3 * intersection_loss + 0 * distortion_loss + 5e3 * d_grid
            val_repr_loss += reprojection_loss.item()
            val_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()  
            distort_loss += distortion_loss.item()  
            grid_norm_loss += d_grid.item()  
            total_loss += loss.item()
    return total_loss / len(val_loader.dataset), val_repr_loss / len(val_loader.dataset), val_dist_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset), intersection_loss / len(val_loader.dataset), distort_loss / len(val_loader.dataset), grid_norm_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_real_two_cams.flatten(1), pixels_virtual_two_cams.flatten(1))
_, _, _, recon_loss_init, _, _ = arena(pixels_real_two_cams.flatten(1), pixels_virtual_two_cams.flatten(1))
print(f'Initialization loss: {recon_loss_init.mean()}')
plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=12
rand_ind = torch.randperm(virtual_pixels_cam_0.shape[-1])
test_dataset_size = 2
pixels_virtual_two_cams_train_val = pixels_virtual_two_cams[...,rand_ind[test_dataset_size:]]
pixels_real_two_cams_train_val = pixels_real_two_cams[..., rand_ind[test_dataset_size:]]
target_coordinates_train_val = target_coordinates[..., rand_ind[test_dataset_size:]]
grid = grid.repeat(1,1,target_coordinates_train_val.shape[-1])
pixels_virtual_two_cams_test = pixels_virtual_two_cams[..., rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[..., rand_ind[:test_dataset_size]]
pixels_real_two_cams_test = pixels_real_two_cams[..., rand_ind[:test_dataset_size]]

dataset = CalibrationDataset(pixels_virtual_two_cams_train_val, pixels_real_two_cams_train_val, target_coordinates_train_val, grid)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 1000
optimizer = optim.Adam(arena.parameters(), lr=1e-2
                       )
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)
pixels_virtual_two_cams = pixels_virtual_two_cams.to(device)


# %% Training loop
gt_train_loss_array = []
closest_distance_train_loss_array = []
gt_val_loss_array = []
closest_distance_val_loss_array = []
best_loss = 1e100

#%%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
plot = False
for epoch in tqdm(range(num_epochs)):
    train_loss, repr_loss, dist_loss, triangulation_loss, intersection_loss, distortion_loss, grid_norm_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    
    if epoch == 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-3

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4
    
    if epoch == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4

    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train loss: {train_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, intersection_loss: {intersection_loss}, distortion loss : {distortion_loss}, grid norm: {grid_norm_loss}')
        if epoch % 200 == 0:
            plot = True
    gt_train_loss_array.append(repr_loss)
    closest_distance_train_loss_array.append(dist_loss)
    writer.add_scalar('Loss/Train_total', train_loss, epoch)

    val_loss, repr_loss, dist_loss, triangulation_loss, intersection_loss, distortion_loss, grid_norm_loss = validate(model=arena,
             val_dataloader=val_loader,
             criterion=criterion,
             )

    writer.add_scalar('Loss/Val_total', val_loss, epoch)
    writer.add_scalar('Loss/Val_Reprojection_error', repr_loss, epoch)
    writer.add_scalar('Loss/Val_Closest_distance_error', dist_loss, epoch)
    writer.add_scalar('Loss/Val_Triangulation_error', triangulation_loss, epoch)
    writer.add_scalar('Loss/Val_Grid_norm_error', grid_norm_loss, epoch)
    writer.add_scalar('Loss/Val_Intersection_error', intersection_loss, epoch)
    writer.add_scalar('Loss/Val_Distortion_error', distortion_loss, epoch)
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
    
    #scheduler.step(val_loss)
    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, intersection_loss: {intersection_loss}, distortion loss: {distortion_loss}, grid norm: {grid_norm_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if 1e1*repr_loss + 1e2*dist_loss + 1e3 * intersection_loss + 5e3 * grid_norm_loss < best_loss:
        best_loss = 1e1*repr_loss + 1e2*dist_loss + 1e3 * intersection_loss + 5e3 * grid_norm_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')
        print(f'Found better model with validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}')
    gt_val_loss_array.append(repr_loss)
    closest_distance_val_loss_array.append(dist_loss)


 # %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])

recon_3D_test, closest_dist_test, recon_real_1, reprojection_error_test, intersection_penalty, distortion_loss_test = arena(pixels_real_two_cams_test.flatten(1), pixels_virtual_two_cams_test.flatten(1))
pixels_real_two_cam_0_test_ = pixels_real_two_cams_test[:2,:].flatten(1)
test_loss = euclidean_distance(
                            target_coordinates_test.flatten(1), recon_3D_test
                            )

print(f'Triangulation loss: {test_loss.mean()}')
print(f'Closest distance loss: {closest_dist_test.mean()}')
print(f'Distortion loss: {distortion_loss_test.mean()}')

reprojection_error_1 = torch.linalg.norm(
                                    recon_real_1 - pixels_real_two_cam_0_test_, dim=0
                                    )
print(f'Reprojection error: cam_0 : {reprojection_error_1.mean()}')

plt.figure(figsize=(15,15))
plt.scatter(
    recon_real_1[0,:].detach().numpy(),
    recon_real_1[1,:].detach().numpy(),
    color='g',
    marker='o',
    label='Estimate',
)
plt.scatter(
    pixels_real_two_cam_0_test_[0,:].detach().numpy(),
    pixels_real_two_cam_0_test_[1,:].detach().numpy(),
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
ax.set_title(f'Reprojection error: {(reprojection_error_1.mean()):.2f} pixels', fontsize=22)
plt.legend(fontsize=22)
plt.savefig(f'{model_checkpoint_dir}/reprojection_error.png')


#%% Test loss
#%% Make plots after training
plt.figure(figsize=(15,15))
epochs = np.arange(0, len(gt_train_loss_array), 1)
plt.loglog(epochs, gt_train_loss_array, color='b', 
         label='Ground truth train loss')
plt.loglog(epochs, closest_distance_train_loss_array, 
         color='r', 
         label='Closest distance train loss')
plt.loglog(epochs, gt_val_loss_array, color='b', 
         label='Ground truth val loss',
         linestyle='--')
plt.loglog(epochs, closest_distance_val_loss_array, 
         color='r', 
         label='Closest distance val loss',
         linestyle='--')
plt.legend(fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss (pixel)', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/training_loss.png')

#%%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    target_coordinates_test[0,:].detach().numpy(),
    target_coordinates_test[1,:].detach().numpy(),
    target_coordinates_test[2,:].detach().numpy(),
    color='r',
    label='Ground truth',
)
ax.scatter(
    recon_3D_test[0,:].detach().numpy(),
    recon_3D_test[1,:].detach().numpy(),
    recon_3D_test[2,:].detach().numpy(),
    color='g',
    label='Estimate',
)
ax.set_xlabel('X (mm)', fontsize=22)
ax.set_ylabel('Y (mm)', fontsize=22)
ax.set_zlabel('Z (mm)', fontsize=22)
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
ax.set_zticklabels(ax.get_zticks(), fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/3D_scatter.png')

#%% Visualize trained arena
arena.visualize(pixels_real_two_cams_test, pixels_virtual_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')

# %%
