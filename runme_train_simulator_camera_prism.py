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
#from torchsummary import summary
from torch.utils.data import DataLoader, random_split, Dataset
pi = torch.tensor(np.pi, dtype=torch.float64)
torch.autograd.set_detect_anomaly(True)
import datetime
import time
from arenas.prism_arenas import Arena_reprojection_loss_two_cameras
from utils import euclidean_distance


#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, data, labels_2D, labels_3D):
        # Example data
        self.data = data.T
        self.labels_2D = labels_2D.T
        self.labels_3D = labels_3D.T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels_2D[idx], self.labels_3D[idx]


#%% Load camera calibration results
load_checkpoint = False
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_2/results/'
calibration_results_file = 'dotted_grid_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02'], dtype=torch.float64).T - 1.
virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1'], dtype=torch.float64).T - 1.
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float64).T

stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

focal_length_cam_1 = (K1[0,0] + K1[1,1]) /  2
focal_length_cam_2 = (K2[0,0] + K2[1,1]) /  2



"""
principal_point_pixel_cam_0 = [638.040 - 1, 492.499 - 1] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778 - 1, 521.5078 - 1] # This comes from the calibration results


R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T.to(torch.float64)
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None].to(torch.float64)
focal_length_cam_1 = 5696.3 # in pixels
focal_length_cam_2 = 5709.3 # in pixels
"""

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

#%% Prism corners third plane 
prism_annotated_face = 'first'
prism_corners_path = f'{calibration_results_dir}/prism_corners_{prism_annotated_face}_plane.mat'
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

if prism_annotated_face == 'third':
    prism3_axes[:,0] = torch.linalg.cross(prism3_axes[:,1], prism3_axes[:,2])
    prism3_axes = prism3_axes / torch.linalg.norm(prism3_axes, dim=0)
    prism1_axes = torch.mm(rotx(pi/2), prism3_axes)

    prism_a = 20.
    prism_b = 20.
    prism3_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0)
    prism1_center = prism3_center + prism_b/2 * prism1_axes[:,0] - prism_b/2 * prism1_axes[:,2] 

elif prism_annotated_face == 'first':
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
    prism1_center = torch.tensor(prism_corners, dtype=torch.float64).mean(dim=0)
    prism1_axes[:,0] = torch.linalg.cross(prism1_axes[:,1], prism1_axes[:,2])
    prism1_axes = prism1_axes / torch.linalg.norm(prism1_axes, dim=0)

plane = Plane(axes=prism1_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma], dtype=torch.float64)


#%% Initialize an Arena instance
prism_distance = torch.tensor(130.) # Not used if you're using fiduciary markers for initialization
#prism_angles = torch.tensor([-1.5341, 1.3650, -1.5638], dtype=torch.float64)
#prism_center = torch.tensor([-6.5058, -5.6897, 149.0993], dtype=torch.float64)
arena = Arena_reprojection_loss_two_cameras(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            prism_angles=prism_angles,
            prism_center=prism1_center)
pixels_virtual_two_cams = torch.vstack((virtual_pixels_cam_0, virtual_pixels_cam_1))
pixels_real_two_cams = torch.vstack((undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1))
freeze_camera_parameters(arena.camera1)
freeze_stereocamera(arena)

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
    
    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for input, label_2D, label_3D in train_loader:         
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            output = model(
                input.T,
                label_2D.T)
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_distorted_virtual_pixels_1, recon_distorted_virtual_pixels_2, recon_distorted_real_pixels_1, recon_distorted_real_pixels_2, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2 = output['recon_3D'], output['closest_distance_mean'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_distorted_virtual_pixels_1'], output['recon_distorted_virtual_pixels_2'], output['recon_distorted_real_pixels_1'], output['recon_distorted_real_pixels_2'], output['dist_penalty_1'], output['dist_penalty_2'], output['int_penalty_1'], output['int_penalty_2']
            
            
            if plot:                
                rand_ind = torch.randperm(recon_distorted_virtual_pixels_1.shape[1])
                plt.subplot(121)
                plt.scatter(
                    recon_distorted_virtual_pixels_1[0,rand_ind].detach().numpy(),
                    recon_distorted_virtual_pixels_1[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    label_2D.T[0,rand_ind].detach().numpy(),
                    label_2D.T[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='g',
                )
                plt.subplot(122)
                plt.scatter(
                    recon_distorted_virtual_pixels_2[0,rand_ind].detach().numpy(),
                    recon_distorted_virtual_pixels_2[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    label_2D.T[2,rand_ind].detach().numpy(),
                    label_2D.T[3,rand_ind].detach().numpy(),
                    s=0.5,
                    c='g',
                )

            # Ground truth pixel error calculated using triangulation of refracted pixels
            recon_virtual_loss = (euclidean_distance(
                recon_distorted_virtual_pixels_1,
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_distorted_virtual_pixels_2, 
                label_2D[:,2:].T).sum()) / 2

            recon_real_loss = (euclidean_distance(
                recon_distorted_real_pixels_1, 
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_distorted_real_pixels_2, 
                label_2D[:,2:].T).sum()) / 2
            
            overall_reprojection_loss = (euclidean_distance(
                recon_pixels_1, 
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_pixels_2,
                label_2D[:,2:].T).sum()) / 2

            triangulation_loss = euclidean_distance(
                                    recon_3D, label_3D.T
                                ).sum()
            
            distortion_loss = dist_penalty_1.sum() + dist_penalty_2.sum()
            intersection_loss = int_penalty_1.sum() + int_penalty_2.sum()
            closest_distance_loss = closest_distance.sum()
            tr_loss += triangulation_loss.item()

            # Recon_real_loss is the pixel error between the reprojected 3D point from real pixels and the real pixel
            loss = overall_reprojection_loss + 0*recon_real_loss + 0*recon_virtual_loss + (1e2 * intersection_loss) + distortion_loss + closest_distance_loss + 2 * triangulation_loss 
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            virtual_loss += recon_virtual_loss.item()
            real_loss += recon_real_loss.item()
            closest_dist_loss += closest_distance_loss.item()
            distortion_loss += distortion_loss.item()
            intersection_loss += intersection_loss.item()
            reprojection_loss += overall_reprojection_loss.item()        
    return total_loss / len(train_loader.dataset), reprojection_loss / len(train_loader.dataset), virtual_loss / len(train_loader.dataset), real_loss / len(train_loader.dataset), closest_dist_loss / len(train_loader.dataset), distortion_loss / len(train_loader.dataset), intersection_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset)


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
    with torch.no_grad():
        for input, label_2D, label_3D in val_loader:
            output = model(input.T, label_2D.T)
            recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_distorted_virtual_pixels_1, recon_distorted_virtual_pixels_2, recon_distorted_real_pixels_1, recon_distorted_real_pixels_2, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2 = output['recon_3D'], output['closest_distance_mean'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_distorted_virtual_pixels_1'], output['recon_distorted_virtual_pixels_2'], output['recon_distorted_real_pixels_1'], output['recon_distorted_real_pixels_2'], output['dist_penalty_1'], output['dist_penalty_2'], output['int_penalty_1'], output['int_penalty_2']
            recon_pixels = torch.cat((recon_distorted_virtual_pixels_1,
            recon_distorted_virtual_pixels_2), dim=0)

            recon_virtual_loss = (euclidean_distance(
                recon_distorted_virtual_pixels_1, 
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_distorted_virtual_pixels_2, 
                label_2D[:,2:].T).sum()) / 2
            closest_distance_loss = closest_distance.sum()

            recon_real_loss = (euclidean_distance(
                recon_distorted_real_pixels_1, 
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_distorted_real_pixels_2, 
                label_2D[:,2:].T).sum()) / 2
            
            overall_reprojection_loss = (euclidean_distance(
                recon_pixels_1, 
                label_2D[:,:2].T).sum() + euclidean_distance(
                recon_pixels_2,
                label_2D[:,2:].T).sum()) / 2
            
            triangulation_loss = euclidean_distance(
                                    recon_3D, label_3D.T
                                ).sum()

            distortion_loss = dist_penalty_1.sum() + dist_penalty_2.sum()
            intersection_loss = int_penalty_1.sum() + int_penalty_2.sum()
            closest_distance_loss = closest_distance.sum()

            loss = overall_reprojection_loss + 0*recon_real_loss + 0*recon_virtual_loss + 1e2 * intersection_loss + distortion_loss + closest_distance_loss + 0 * triangulation_loss 

            total_loss += loss.item()
            virtual_loss += recon_virtual_loss.item()
            real_loss += recon_real_loss.item()
            intersection_loss += intersection_loss.item()
            distortion_loss += distortion_loss.item()
            closest_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
            reprojection_loss += overall_reprojection_loss.item()

    return total_loss / len(val_loader.dataset), reprojection_loss / len(val_loader.dataset), virtual_loss / len(val_loader.dataset), real_loss / len(val_loader.dataset), closest_dist_loss / len(val_loader.dataset), distortion_loss / len(val_loader.dataset), intersection_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_virtual_two_cams)
plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1024
rand_ind = torch.randperm(pixels_virtual_two_cams.shape[1])
test_dataset_size = 150
pixels_virtual_two_cams_test = pixels_virtual_two_cams[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]
pixels_real_two_cams_test = pixels_real_two_cams[:, rand_ind[:test_dataset_size]]

pixels_virtual_two_cams = pixels_virtual_two_cams[:,rand_ind[test_dataset_size:]]
pixels_real_two_cams = pixels_real_two_cams[:, rand_ind[test_dataset_size:]]
target_coordinates = target_coordinates[:, rand_ind[test_dataset_size:]]

dataset = CalibrationDataset(pixels_virtual_two_cams, pixels_real_two_cams, target_coordinates)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_virtual_two_cams_train, pixels_virtual_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_virtual_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_virtual_two_cams_val, batch_size=batch_size, shuffle=False)

num_epochs = 1000
optimizer = optim.Adam(arena.parameters(), lr=1e-2
                       )
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
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
for epoch in tqdm(range(num_epochs)):
    if epoch == 300:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-3
    if epoch == 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2e-4
    
    train_loss, train_reprojection_loss, train_virtual_loss, train_real_loss, train_closest_dist_loss, train_distortion_loss, train_intersection_loss, triangulation_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    
    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train_loss: {train_loss}, reprojection loss: {train_reprojection_loss}, closest_dist_loss: {train_closest_dist_loss}, triangulation loss : {triangulation_loss}')
        if epoch % 200 == 0:
            plot = True
    train_loss_array.append(train_loss)
    train_virtual_loss_array.append(train_virtual_loss)
    train_real_loss_array.append(train_real_loss)
    train_closest_distance_loss_array.append(train_closest_dist_loss)
    train_distortion_loss_array.append(train_distortion_loss)
    train_intersection_loss_array.append(train_intersection_loss)

    val_loss, val_reprojection_loss, val_virtual_loss, val_real_loss, val_closest_dist_loss, val_distortion_loss,val_intersection_loss, triangulation_loss = validate(model=arena,
             val_loader=val_loader,
             criterion=criterion,
             )
    
    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: val_loss: {val_loss}, reprojection error: {val_reprojection_loss}, closest_dist_loss: {val_closest_dist_loss}, triangulation loss {triangulation_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')
        print(f'Found better model with validation loss for epoch {epoch}: val_loss: {val_loss}, reprojection error: {val_reprojection_loss}, closest_dist_loss: {val_closest_dist_loss}, triangulation loss {triangulation_loss}')
    gt_train_loss_array.append(train_loss)
    gt_val_loss_array.append(val_loss)



# %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

output = arena(pixels_virtual_two_cams_test, pixels_real_two_cams_test)
recon_3D_test, closest_dist_test, recon_pixels_1, recon_pixels_2, recon_distorted_virtual_pixels_1, recon_distorted_virtual_pixels_2, recon_distorted_real_pixels_1, recon_distorted_real_pixels_2, dist_penalty_1, dist_penalty_2, int_penalty_1, int_penalty_2 = output['recon_3D'], output['closest_distance_mean'], output['recon_pixels_1'], output['recon_pixels_2'], output['recon_distorted_virtual_pixels_1'], output['recon_distorted_virtual_pixels_2'], output['recon_distorted_real_pixels_1'], output['recon_distorted_real_pixels_2'], output['dist_penalty_1'], output['dist_penalty_2'], output['int_penalty_1'], output['int_penalty_2']
triangulation_loss = euclidean_distance(
    recon_3D_test, target_coordinates_test
).mean()

recon_real_loss = (euclidean_distance(
                recon_pixels_1, 
                pixels_real_two_cams_test[:2,:]).mean() + euclidean_distance(
                recon_pixels_2, 
                pixels_real_two_cams_test[2:,:]).mean()) / 2

print(f'Triangulation loss: {triangulation_loss}')
print(f'Distortion penalty: {(dist_penalty_1 + dist_penalty_2).mean()}')
print(f'Real pixel loss: {recon_real_loss.mean()}')

reprojection_loss_1 = euclidean_distance(
    recon_pixels_1, pixels_real_two_cams_test[:2,:]
)
reprojection_loss_2 = euclidean_distance(
    recon_pixels_2, pixels_real_two_cams_test[2:,:]
)

print(f'Reprojection error for two cameras: {reprojection_loss_1.mean()}, {reprojection_loss_2.mean()}')

plt.figure(figsize=(15,15))
plt.scatter(
    recon_distorted_virtual_pixels_1[0,:].detach().numpy(),
    recon_distorted_virtual_pixels_1[1,:].detach().numpy(),
    color='g',
    marker='o',
    label='Estimate',
)
plt.scatter(
    pixels_real_two_cams_test[0,:].detach().numpy(),
    pixels_real_two_cams_test[1,:].detach().numpy(),
    color='r',
    marker='x',
    label='Ground truth',
)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('X (pixels)', fontsize=18)
ax.set_ylabel('Y (pixels)', fontsize=18)
ax.set_title(f'Reprojection error: {reprojection_loss_1.mean():.2f} pixels', fontsize=16)
plt.legend(fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/reprojection_loss.png')



#%% Test loss
#%% Make plots after training
plt.figure(figsize=(15,15))
epochs = np.arange(0,num_epochs)
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
plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss (mm)', fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/training_loss.png')


#%% Visualize trained arena
arena.visualize(pixels_virtual_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
