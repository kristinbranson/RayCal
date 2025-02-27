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
import arenas.fish_tank_arenas.Arena_fish_tank as Arena_fish_tank
from utils import euclidean_distance


#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, data):
        # Example data
        self.data = data.T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#%% Load camera calibration results
load_checkpoint = False
calibration_results_dir = '/groups/branson/bransonlab/aniket/camera_alignment/calibration_new_grid/'
calibration_results_file = 'calibration_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
pixels_two_cams_0 = torch.swapaxes(
    torch.tensor(mat['imagePoints_a'], dtype=torch.float64).T,
     0, 1) - 1.
pixels_two_cams_1 = torch.swapaxes(
    torch.tensor(mat['imagePoints_b'], dtype=torch.float64).T,
     0, 1) - 1.
pixels_two_cams = torch.vstack(
    (pixels_two_cams_0, pixels_two_cams_1)
)
pixels_two_cams = pixels_two_cams.flatten(1)

output_cam_0_pairwise = torch.tensor(mat['imagePoints_a'], dtype=torch.float64).T - 1.
output_cam_1_pairwise = torch.tensor(mat['imagePoints_b'], dtype=torch.float64).T - 1.


stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

focal_length_cam_1 = (K1[0,0] + K1[1,1]) /  2
focal_length_cam_2 = (K2[0,0] + K2[1,1]) /  2


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

def freeze_tank_orientation(arena):
    arena.tank_angles.requires_grad = False

def unfreeze_all_parameters(arena):
    for param in arena.parameters():
        param.requires_grad = True

#%% Initialize an Arena instance
refractive_index_acrylic = torch.tensor([1.48], dtype=torch.float64)
refractive_index_water = torch.tensor([1.33], dtype=torch.float64)
tank_axes = torch.tensor([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
], 
dtype=torch.float64
)
plane_temp = Plane(axes=tank_axes)
tank_angles = torch.tensor(
    [plane_temp.alpha, plane_temp.beta, plane_temp.gamma],
    dtype=torch.float64,
)
tank_center = torch.tensor(
    [50., 0., 250.],
    dtype=torch.float64,
)

tank_size = torch.tensor(
    [300., 500., 130.],
    dtype=torch.float64,
)

tank_thickness = torch.tensor(
    [20.],
    dtype=torch.float64
)

arena = Arena_fish_tank(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            tank_angles=tank_angles,
            tank_center=tank_center,
            tank_size=tank_size,
            tank_thickness=tank_thickness,
            refractive_index_acrylic=refractive_index_acrylic,
            refractive_index_water=refractive_index_water)

freeze_camera_parameters(arena.camera1)
freeze_stereocamera(arena)
freeze_tank_orientation(arena)

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
        for input in train_loader:         
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            recon_3D, closest_distance = model(
                input.T)
            
            closest_distance_loss = closest_distance.sum()

            # Recon_real_loss is the pixel error between the reprojected 3D point from real pixels and the real pixel
            loss = closest_distance_loss  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            closest_dist_loss += closest_distance_loss.item()
            
    return total_loss / len(train_loader.dataset), 


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
        for input in val_loader:
            recon_3D, closest_distance = model(input.T)
            closest_distance_loss = closest_distance.sum()

            loss = closest_distance_loss 
            total_loss += loss.item()
            closest_dist_loss += closest_distance_loss.item()
            
    return total_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_two_cams)
plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1024
rand_ind = torch.randperm(pixels_two_cams.shape[1])
test_dataset_size = 150
pixels_two_cams_test = pixels_two_cams[:, rand_ind[:test_dataset_size]]
pixels_two_cams = pixels_two_cams[:,rand_ind[test_dataset_size:]]

dataset = CalibrationDataset(pixels_two_cams)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_two_cams_train, pixels_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_two_cams_val, batch_size=batch_size, shuffle=False)

num_epochs = 1000
optimizer = optim.Adam(arena.parameters(), lr=1e-1
                       )
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)
pixels_virtual_two_cams = pixels_two_cams.to(device)

# %% Training loop
train_loss_array = []
val_loss_array = []
gt_train_loss_array = []
closest_distance_train_loss_array = []
gt_val_loss_array = []
closest_distance_val_loss_array = []
best_loss = 1e100

plot = False
for epoch in tqdm(range(num_epochs)):
    if epoch == 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-2
    if epoch == 750:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
            unfreeze_all_parameters(arena)
    
    train_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    
    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train_loss: {train_loss}')
        if epoch % 200 == 0:
            plot = True
    train_loss_array.append(train_loss)
    

    val_loss = validate(model=arena,
             val_loader=val_loader,
             criterion=criterion,
             )
    
    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: val_loss: {val_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')
        #print(f'Found better model with validation loss for epoch {epoch}: val_loss: {val_loss}')
    gt_train_loss_array.append(train_loss)
    gt_val_loss_array.append(val_loss)


# %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

recon_3D_test, closest_distance_loss_test = arena(pixels_two_cams_test_)
print(f'Closest distance loss: {closest_distance_loss_test.mean()}')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(
    recon_3D_test[0,:].detach().numpy(),
    recon_3D_test[1,:].detach().numpy(),
    recon_3D_test[2,:].detach().numpy(),
    color='g',
    s=5,
    label='Estimate',
)
ax.set_aspect('equal')


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
arena.visualize(pixels_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
