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
from arenas.prism_arenas import Arena_single_camera_prism_grid_distance
from utils import euclidean_distance
from torch.utils.tensorboard import SummaryWriter


#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, labels_virtual_2D, labels_real_2D, labels_3D, pairwise_distance):
        # Example data
        self.labels_virtual_2D = labels_virtual_2D.T
        self.labels_real_2D = labels_real_2D.T
        self.labels_3D = labels_3D.T
        self.pairwise_distance = pairwise_distance

    def __len__(self):
        return len(self.labels_virtual_2D)

    def __getitem__(self, idx):
        return self.labels_virtual_2D[idx], self.labels_real_2D[idx], self.labels_3D[idx], self.pairwise_distance[idx]

# (1018.5318772136955, 509.0631641086186)
#%% Load camera calibration results
load_checkpoint = False
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_17/results/'
calibration_results_file = 'dotted_grid_pairwise_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
prism_initialization_path = os.path.join(calibration_results_dir, 'prism_initialization.mat')
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
if 'output_data_cam_02_pairwise' in mat.keys():
    virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_pairwise'], dtype=torch.float64).T - 1.
if 'output_data_cam_13_pairwise' in mat.keys():
    virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_pairwise'], dtype=torch.float64).T - 1.
if 'output_data_cam_0_pairwise' in mat.keys():
    real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_pairwise'], dtype=torch.float64).T - 1.
if 'output_data_cam_1_pairwise' in mat.keys():
    undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_pairwise'], dtype=torch.float64).T - 1.
if 'worldPoints_pairwise' in mat.keys():
    target_coordinates = torch.tensor(mat['worldPoints_pairwise'], dtype=torch.float64).T
if 'pairwise_distances' in mat.keys():
    pairwise_distance = torch.tensor(mat['pairwise_distances'][:,0]).to(torch.float64)

if 'stereoParams_export' in mat.keys():  
    stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
#K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
#R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
#T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T + 0.

#principal_point_pixel_cam_0 = torch.tensor([640., 512.]).to(torch.float64) 
#principal_point_pixel_cam_1 = torch.tensor([640., 512.]).to(torch.float64) 

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
#principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

#focal_length_cam_1 = 5206. 
#focal_length_cam_2 = 5206. 

focal_length_cam_0 = (K1[0,0] + K1[1,1]) /  2
#focal_length_cam_2 = (K2[0,0] + K2[1,1]) /  2

# Tensorboard writer
writer = SummaryWriter(log_dir=f'{outputs_dir}/logs/{model_checkpoint_dir}')
print(f'To visualize tensorboard log, run: tensorboard --logdir={outputs_dir}/logs/{model_checkpoint_dir}')


#%% Freeze parameters
def freeze_camera_parameters(camera, distortion=True):
    for param in camera.parameters():
        param.requires_grad = False
    if distortion:
        camera.r1.requires_grad = True

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


def unfreeze_camera_parameters(camera):
    for param in camera.parameters():
        param.requires_grad = True


def unfreeze_all_parameters(arena):
    for param in arena.parameters():
        if not param.requires_grad:
            param.requires_grad = True
    

#%% Prism corners third plane 
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/prism_corners_third_plane.mat'
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_4/results/prism_corners_first_plane.mat'
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

#%% Prism first corner annotations
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/prism_corners_third_plane.mat'
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_4/results/prism_corners_first_plane.mat'
#prism_corners_path = f'{calibration_results_dir}/prism_corners_first_plane.mat'
prism_corners = sio.loadmat(prism_corners_path)['worldPoints']
prism1_axes = torch.zeros(3,3).to(torch.float64)
prism1_axes[:,1] = torch.stack(
    (torch.tensor(prism_corners[1,:] - prism_corners[0,:], dtype=torch.float64),
    )
).mean(dim=0)
prism1_axes[:,2] = torch.stack(
    (torch.tensor(prism_corners[3,:] - prism_corners[0,:], dtype=torch.float64),
     torch.tensor(prism_corners[2,:] - prism_corners[1,:], dtype=torch.float64)
     )
).mean(dim=0)
prism1_axes[:,0] = torch.linalg.cross(prism1_axes[:,1], prism1_axes[:,2])
prism1_axes = prism1_axes / torch.linalg.norm(prism1_axes, dim=0)

prism_a = 20.
prism_b = 20.
prism1_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0)

plane = Plane(axes=prism1_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64)
max_norm = 10.
#prism1_center = torch.tensor([ -5.6856,  -1.2116, 121.2740], dtype=torch.float64)
prism1_center[2] = 120.
prism1_center[1] = -6

#prism1_center[0] = -0
#prism1_center[1] = -0

#%% Initialize an Arena instance
#prism_angles = torch.tensor([-1.8710,  1.4078, -1.9315], dtype=torch.float64)
#plane = Plane(alpha=prism_angles[0], beta=prism_angles[1], gamma=prism_angles[2])
#axes = plane.axes
#axes[1,0] = -axes[1,0]
#axes[0,0] = -axes[0,0]


#plane = Plane(axes=axes)
#prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64)

prism_initializations = sio.loadmat(prism_initialization_path)
prism_center = torch.tensor(prism_initializations['location_prism']).to(torch.float64).T
prism_center[1] = 4.
prism_axes = torch.tensor(prism_initializations['axes_prism']).to(torch.float64)
plane = Plane(axes=prism_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64)

arena = Arena_single_camera_prism_grid_distance(principal_point_pixel_cam_0,
            principal_point_pixel_cam_0, 
            focal_length_cam_0, 
            focal_length_cam_0,
            R_stereo_cam=None,
            T_stereo_cam=None, 
            prism_angles=prism_angles,
            prism_center=prism_center,
            prism_size=prism_a)

freeze_camera_parameters(arena.camera1)
arena.prism.refractive_index_glass.requires_grad = False
arena.prism.prism_size.requires_grad = False

#%% Training and validation functions
def train_two_cams(model, train_loader, criterion, plot=False):    
    model.train()
    repr_loss = 0.
    dist_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.
    distort_loss = 0.

    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for i, (input, label_2D, label_3D, pairwise_distance_batch) in enumerate(train_loader):
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_real_loss, pairwise_distance_recon, intersection_penalty, distortion_penalty = model(label_2D.T, input.T)
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            if epoch == 0 and i == 0:
                print(f'Loss for first iteration: Closest_distance_loss: {closest_distance.mean()}, Pairwise_distance_loss: {torch.abs(d_pairwise_distance).mean()}, reprojection loss: {recon_real_loss.mean()}')
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            intersection_loss = intersection_penalty.sum() / 2
            pairwise_distance_difference = torch.abs(d_pairwise_distance).mean() # Mean absolute difference
            distortion_loss = distortion_penalty.sum() / 2
            
            cam_0_pixel_gt = torch.hstack((label_2D.T[:2,:], label_2D.T[2:4,:]))
            if plot:                
                rand_ind = torch.randperm(label_2D.shape[0])
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
            stacked_label_3D = torch.vstack((label_3D[:,:recon_3D.shape[0]], label_3D[:,recon_3D.shape[0]:]))
            triangulation_loss = euclidean_distance(
                            stacked_label_3D.T, recon_3D
                            ).sum() / 2 # Because there are twice the number of points as the minibatch size
            
            reprojection_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            loss = 1e1 * reprojection_loss  + 0 * triangulation_loss + 1e1 * closest_distance_loss + 5e3 * pairwise_distance_loss + 1e3 * intersection_loss + 0 * distortion_loss
            loss.backward()
            #params_to_clip = [model.prism.prism_angles, model.prism.refractive_index_glass]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            repr_loss += reprojection_loss.item()
            tr_loss += triangulation_loss.item()
            dist_loss += closest_distance_loss.item()  
            pairwise_dist_loss += pairwise_distance_loss.item()   
            distort_loss += distortion_loss.item()       
            total_loss += loss.item()
    return total_loss / len(train_loader.dataset), repr_loss / len(train_loader.dataset), dist_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset), pairwise_dist_loss / len(train_loader.dataset),  intersection_loss / len(train_loader.dataset), distort_loss / len(train_loader.dataset)


def validate(model, val_dataloader, criterion):
    model.eval()
    val_repr_loss = 0.
    val_dist_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.
    distort_loss = 0.
    with torch.no_grad():
        for input, label_2D, label_3D, pairwise_distance_batch in val_dataloader:
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_real_loss, pairwise_distance_recon, intersection_penalty, distortion_penalty = model(label_2D.T, input.T)
            reprojection_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            intersection_loss = intersection_penalty.sum() / 2
            stacked_label_3D = torch.vstack((label_3D[:,:recon_3D.shape[0]], label_3D[:,recon_3D.shape[0]:]))
            triangulation_loss = euclidean_distance(
                            stacked_label_3D.T, recon_3D
                            ).sum() / 2 # Because there are twice the number of points as the minibatch size
            distortion_loss = distortion_penalty.sum() / 2
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            loss = 1e1 * recon_real_loss.sum()  + 0 * triangulation_loss + 1 * closest_distance_loss + 5e3 * pairwise_distance_loss + 1e3 * intersection_loss + 0 * distortion_loss
            val_repr_loss += reprojection_loss.item()
            val_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
            pairwise_dist_loss += pairwise_distance_loss.item()     
            distort_loss += distortion_loss.item()    
            total_loss += loss.item()
    return total_loss / len(val_loader.dataset), val_repr_loss / len(val_loader.dataset), val_dist_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset), pairwise_dist_loss / len(val_loader.dataset), intersection_loss / len(val_loader.dataset), distort_loss / len(val_loader.dataset)

#%% Visualize arena initialization
arena.visualize(real_pixels_cam_0, virtual_pixels_cam_0, color_labels=True)
_, _, _, recon_loss_init, pairwise_distance_recon, _, _ = arena(real_pixels_cam_0, virtual_pixels_cam_0)
pairwise_distance_loss = torch.abs((pairwise_distance_recon.detach() - pairwise_distance))
print(f'Initialization loss: {recon_loss_init.mean()}, pairwise_distance_loss: {pairwise_distance_loss.mean()}')
plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1028
rand_ind = torch.randperm(virtual_pixels_cam_0.shape[1])
test_dataset_size = 150
virtual_pixels_cam_0_train_val = virtual_pixels_cam_0[:,rand_ind[test_dataset_size:]]
real_pixels_cam_0_train_val = real_pixels_cam_0[:, rand_ind[test_dataset_size:]]
target_coordinates_train_val = target_coordinates[:, rand_ind[test_dataset_size:]]
pairwise_distance_train_val = pairwise_distance[rand_ind[test_dataset_size:]]
virtual_pixels_cam_0_test = virtual_pixels_cam_0[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]
real_pixels_cam_0_test = real_pixels_cam_0[:, rand_ind[:test_dataset_size]]
pairwise_distance_test = pairwise_distance[rand_ind[:test_dataset_size]]

dataset = CalibrationDataset(virtual_pixels_cam_0_train_val, real_pixels_cam_0_train_val, target_coordinates_train_val, pairwise_distance_train_val)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 3000
lr = 1e-2
optimizer = optim.Adam(arena.parameters(), lr=lr
                       )

#optimizer = optim.Adam(params)
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)
virtual_pixels_cam_0 = virtual_pixels_cam_0.to(device)


# %% Training loop
gt_train_loss_array = []
closest_distance_train_loss_array = []
gt_val_loss_array = []
closest_distance_val_loss_array = []
best_loss = 1e100

#%%
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
plot = False
for epoch in tqdm(range(num_epochs)):
    train_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss, intersection_loss, distortion_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-3

    if epoch == 150:        
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3

    if epoch == 500:
        for param_group in optimizer.param_groups:            
            param_group['lr'] = 5e-4
        unfreeze_all_parameters(arena)

    if epoch == 650:
        for param_group in optimizer.param_groups:            
            param_group['lr'] = 1e-4
    

    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train loss: {train_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance_loss: {pairwise_distance_loss}, intersection_loss: {intersection_loss}, distortion loss : {distortion_loss}')
        if epoch % 200 == 0:
            plot = True
    gt_train_loss_array.append(repr_loss)
    closest_distance_train_loss_array.append(dist_loss)
    writer.add_scalar('Loss/train', train_loss, epoch)

    val_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss, intersection_loss, distortion_loss = validate(model=arena,
             val_dataloader=val_loader,
             criterion=criterion,
             )
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/val_reprojection_error', repr_loss, epoch)
    writer.add_scalar('Loss/val_closest_distance_error', dist_loss, epoch)
    writer.add_scalar('Loss/val_triangulation_error', triangulation_loss, epoch)
    writer.add_scalar('Loss/val_pairwise_distance_error', pairwise_distance_loss, epoch)
    writer.add_scalar('Loss/val_intersection_error', intersection_loss, epoch)
    writer.add_scalar('Loss/val_distortion_error', distortion_loss, epoch)
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
        print(f'Validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance loss: {pairwise_distance_loss}, intersection_loss: {intersection_loss}, distortion loss: {distortion_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if 1e1*repr_loss + 1e2*dist_loss + 5e3*pairwise_distance_loss + 1e3*intersection_loss < best_loss:
        best_loss = 1e1*repr_loss + 1e2*dist_loss + 5e3*pairwise_distance_loss + 1e3*intersection_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')
        print(f'Found better model with validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance loss: {pairwise_distance_loss}')
    gt_val_loss_array.append(repr_loss)
    closest_distance_val_loss_array.append(dist_loss)
writer.close()


 # %% Validate the model
PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
target_coordinates_test_ = torch.hstack((target_coordinates_test[:3,:], target_coordinates_test[3:,:]))
recon_3D_test, closest_dist_test, recon_real_1, reprojection_error_test, pairwise_distance_test_, intersection_penalty, distortion_loss_test = arena(real_pixels_cam_0_test, virtual_pixels_cam_0_test)
pixels_real_two_cam_0_test_ = torch.hstack((real_pixels_cam_0_test[:2,:], real_pixels_cam_0_test[2:4,:]))
test_loss = euclidean_distance(
                            target_coordinates_test_, recon_3D_test
                            )
pairwise_distance_loss = torch.abs(pairwise_distance_test_ - pairwise_distance_test).mean()
print(f'Pairwise distance loss: {pairwise_distance_loss.mean()}')
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
    target_coordinates_test_[0,:].detach().numpy(),
    target_coordinates_test_[1,:].detach().numpy(),
    target_coordinates_test_[2,:].detach().numpy(),
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

#%%
reprojection_error_1 = reprojection_error_1.detach().numpy()
import matplotlib.ticker as ticker
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
Q1 = np.percentile(reprojection_error_1, 25)
Q3 = np.percentile(reprojection_error_1, 75)
IQR = Q3 - Q1
n = len(reprojection_error_1)

# Calculate bin width using Freedman-Diaconis rule
bin_width = 2 * IQR / (n ** (1/3))

# Calculate number of bins
data_range = np.max(reprojection_error_1) - np.min(reprojection_error_1)
num_bins = int(np.ceil(data_range / bin_width))
ax.hist(reprojection_error_1, bins=num_bins, edgecolor='black')
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set_xlabel('Reprojection error (pixels)', fontsize=28)

#%% Visualize trained arena
fig, ax = arena.visualize(real_pixels_cam_0_test, virtual_pixels_cam_0_test, color_labels=True)

#%%
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')


# %%
