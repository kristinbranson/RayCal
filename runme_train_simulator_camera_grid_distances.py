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
from arenasEfficient import Arena_two_real_cameras
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
calibration_results_file = 'dotted_grid_pairwise_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_pairwise'], dtype=torch.float64, requires_grad=True).T - 1.
virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_pairwise'], dtype=torch.float64, requires_grad=True).T - 1.
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_pairwise'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_pairwise'], dtype=torch.float64).T - 1.
target_coordinates = torch.tensor(mat['worldPoints_pairwise'], dtype=torch.float64).T
stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T + 10.

principal_point_pixel_cam_0 = torch.tensor([640., 512.]).to(torch.float64) #torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([640., 512.]).to(torch.float64) #torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

focal_length_cam_1 = 5208. #(K1[0,0] + K1[1,1]) /  2
focal_length_cam_2 = 5208. #(K2[0,0] + K2[1,1]) /  2

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

#%% Prism corners third plane 
prism_corners_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/prism_corners_third_plane.mat'
prism_corners_path = f'{calibration_results_dir}/prism_corners_first_plane.mat'
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



#%% Initialize an Arena instance
prism_distance = torch.tensor(130.) # Not used if you're using fiduciary markers for initialization
arena = Arena_two_real_cameras(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            prism_angles=prism_angles,
            prism_center=prism1_center)
pixels_virtual_two_cams = torch.vstack((virtual_pixels_cam_0, virtual_pixels_cam_1))
pixels_real_two_cams = torch.vstack((undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1))


#%% Training and validation functions
def train_two_cams(model, train_loader, criterion, plot=False):    
    model.train()
    gt_loss = 0.
    dist_loss = 0.
    tr_loss = 0.
    num_batches = 0
    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for input, label_2D, label_3D in train_loader:         
            num_batches += 1
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_real_loss = model(
                input.T,
                label_2D.T)
            if plot:                
                rand_ind = torch.randperm(recon_distorted_pixels_1.shape[1])
                plt.subplot(121)
                plt.scatter(
                    recon_distorted_pixels_1[0,rand_ind].detach().numpy(),
                    recon_distorted_pixels_1[1,rand_ind].detach().numpy(),
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
                    recon_distorted_pixels_2[0,rand_ind].detach().numpy(),
                    recon_distorted_pixels_2[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='r',
                )
                plt.scatter(
                    label_2D.T[2,rand_ind].detach().numpy(),
                    label_2D.T[3,rand_ind].detach().numpy(),
                    s=0.5,
                    c='g',
                )
            
            triangulation_loss = euclidean_distance(
                            label_3D.T, recon_3D
            ).sum()
            #ground_truth_loss = criterion(output, label.T)
            ground_truth_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            loss = recon_real_loss.sum()  + 5 * triangulation_loss + 5 * closest_distance_loss
            loss.backward()
            optimizer.step()
            gt_loss += ground_truth_loss.item()
            tr_loss += triangulation_loss.item()
            dist_loss += closest_distance_loss.item()            
    return gt_loss / len(train_loader.dataset), dist_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset)


def train_one_cam(model, pixels):
    model.train()
    optimizer.zero_grad()
    output_ray = model(pixels)
    closest_distance_loss = output_ray.distance_to_point(target_coordinates).mean()
    closest_distance_loss.backward()
    optimizer.step()
    return closest_distance_loss.item()

def validate(model, val_dataloader, criterion):
    model.eval()
    val_gt_loss = 0.
    val_dist_loss = 0.
    tr_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for input, label_2D, label_3D in val_dataloader:
            num_batches += 1
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_real_loss = model(input.T, label_2D.T)
            ground_truth_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            triangulation_loss = euclidean_distance(
                            label_3D.T, recon_3D
                            ).sum()
            val_gt_loss += ground_truth_loss.item()
            val_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
    return val_gt_loss / len(val_loader.dataset), val_dist_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_virtual_two_cams)
_, _, _, _, init_loss = arena(pixels_virtual_two_cams, pixels_real_two_cams)
print(f'Initialization loss: {init_loss.mean()}')

plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1024
rand_ind = torch.randperm(pixels_virtual_two_cams.shape[1])
test_dataset_size = 150
pixels_virtual_two_cams_test = pixels_virtual_two_cams[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]
pixels_real_two_cams_test = pixels_real_two_cams[:, rand_ind[:test_dataset_size]]

pixels_virtual_two_cams = pixels_virtual_two_cams[:,test_dataset_size:]
pixels_real_two_cams = pixels_real_two_cams[:, test_dataset_size:]
target_coordinates = target_coordinates[:, test_dataset_size:]

dataset = CalibrationDataset(pixels_virtual_two_cams, pixels_real_two_cams, target_coordinates)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_virtual_two_cams_train, pixels_virtual_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_virtual_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_virtual_two_cams_val, batch_size=batch_size, shuffle=False)

num_epochs = 400
optimizer = optim.Adam(arena.parameters(), lr=5e-1
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
plot = False
for epoch in tqdm(range(num_epochs)):
    gt_loss, dist_loss, triangulation_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    
    if epoch == 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-1

    if epoch == 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-2

    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: gt_loss: {gt_loss}, dist_loss: {dist_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}')
        if epoch % 200 == 0:
            plot = True
    gt_train_loss_array.append(gt_loss)
    closest_distance_train_loss_array.append(dist_loss)

    gt_loss, dist_loss, triangulation_loss = validate(model=arena,
             val_dataloader=val_loader,
             criterion=criterion,
             )
    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: gt_loss: {gt_loss}, dist_loss: {dist_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': gt_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if gt_loss + dist_loss + triangulation_loss < best_loss:
        best_loss = gt_loss + dist_loss + triangulation_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': gt_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')

    gt_val_loss_array.append(gt_loss)
    closest_distance_val_loss_array.append(dist_loss)


# %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

recon_3D_test, closest_dist_test, recon_real_1, recon_real_2, reprojection_error_test = arena(pixels_virtual_two_cams_test, 
                                                        pixels_real_two_cams_test)
test_loss = euclidean_distance(
                            target_coordinates_test, recon_3D_test
                            ).mean()


print(f'Triangulation loss: {test_loss.mean()}')
#cameraMatrix1 = torch.tensor([[5707.2115, 0., 648.0465, 0.], [0, 5707.2115 , 512.79, 0.], [0., 0., 1., 0.]],
#                             dtype=torch.float64)
K1[0,0] = arena.camera1.focal_length_pixels
K1[1,1] = arena.camera1.focal_length_pixels
K1[0,2] = arena.camera1.principal_point_pixel[0]
K1[1,2] = arena.camera1.principal_point_pixel[1]
K1_ = torch.cat((K1, torch.tensor([[0., 0., 0.]]).T.to(torch.float64)), dim=1)
cameraMatrix1 = K1_
homogeneous_recon_3D_test = torch.vstack((recon_3D_test,
                                          torch.zeros(1,recon_3D_test.shape[1], 
                                                      dtype=torch.float64)))
homogeneous_target_coordinates_test = torch.vstack((target_coordinates_test,
                                          torch.zeros(1,target_coordinates_test.shape[1],
                                                      dtype=torch.float64)))
reprojection_pixels_test_homogeneous = cameraMatrix1 @ homogeneous_recon_3D_test
#reprojection_pixels_test = reprojection_pixels_test_homogeneous[:2,:] / reprojection_pixels_test_homogeneous[-1,:][None,:]

reprojection_error_1 = torch.linalg.norm(
                                    recon_real_1 - pixels_real_two_cams_test[:2,:], dim=0
                                    )
reprojection_error_2 = torch.linalg.norm(
                                    recon_real_2 - pixels_real_two_cams_test[2:,:], dim=0
                                    )
print(f'Reprojection error: cam_0 : {reprojection_error_1.mean()}, cam_1 : {reprojection_error_2.mean()}')

plt.figure()
plt.scatter(
    recon_real_1[0,:].detach().numpy(),
    recon_real_1[1,:].detach().numpy(),
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
ax.set_xlabel('X (pixels)', fontsize=22)
ax.set_ylabel('Y (pixels)', fontsize=22)
ax.set_xticklabels(ax.get_xticks(), fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)
ax.set_title(f'Reprojection error: {((reprojection_error_1.mean()+reprojection_error_2.mean())/2):.2f} pixels', fontsize=22)
plt.legend(fontsize=22)
plt.savefig(f'{model_checkpoint_dir}/reprojection_error.png')


#%% Test loss
#%% Make plots after training
plt.figure()
epochs = np.arange(0, num_epochs, 1)
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
plt.ylabel('Loss (mm)', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f'{model_checkpoint_dir}/training_loss.png')


#%% Visualize trained arena
arena.visualize(pixels_virtual_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
