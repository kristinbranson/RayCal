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
from arenas.fish_tank_arenas import Arena_Akihiro_fish_tank_pairwise_distances
from utils import euclidean_distance
from torch.utils.tensorboard import SummaryWriter


#%% Dataloader
class CalibrationDataset(Dataset):
    def __init__(self, data, pairwise_distances):
        # Example data
        self.data = data.T
        self.pairwise_distances = pairwise_distances

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.pairwise_distances[idx]

#%% Load data
load_checkpoint = False
calibration_results_dir = os.path.join(
    '/groups/branson/bransonlab/aniket/camera_alignment/',
    '2025-07-29_danionella_priapus_tank/camera_calibration_2025-07-29/calibration_new_grid'
)

top_whole_grid = torch.tensor(sio.loadmat(os.path.join(
    calibration_results_dir,
    'top.mat'
)
)['calibration_grid_points_top'][0]).T
side1_whole_grid = torch.tensor(sio.loadmat(os.path.join(
    calibration_results_dir,
    'side1.mat'
)
)['calibration_grid_points_side1'][0]).T
side2_whole_grid = torch.tensor(sio.loadmat(os.path.join(
    calibration_results_dir,
    'side2.mat'
)
)['calibration_grid_points_side2'][0]).T


#%%
calibration_results_file = 'results.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_fish'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=f'{outputs_dir}/logs/{model_checkpoint_dir}')

mat = sio.loadmat(calibration_results_path)
output_cam_0_pairwise = torch.tensor(mat['output_cam_0_pairwise'], dtype=torch.float64).T - 1.
output_cam_1_pairwise = torch.tensor(mat['output_cam_2_pairwise'], dtype=torch.float64).T - 1.
output_cam_2_pairwise = torch.tensor(mat['output_cam_1_pairwise'], dtype=torch.float64).T - 1.

max_data_points_to_use = 10000 # Grid point pairs to use for calibration
if output_cam_0_pairwise.shape[1] > max_data_points_to_use:
    num_data_points = max_data_points_to_use

#sampled_data_points = rand_ind = torch.randperm(output_cam_0_pairwise.shape[1])[:num_data_points]
rand_ind = output_cam_0_pairwise[:num_data_points]
output_cam_0_pairwise = output_cam_0_pairwise[:,:num_data_points]
output_cam_1_pairwise = output_cam_1_pairwise[:,:num_data_points]
output_cam_2_pairwise = output_cam_2_pairwise[:,:num_data_points]

pixels_all_cams = torch.vstack((output_cam_0_pairwise,
                                output_cam_1_pairwise,
                                output_cam_2_pairwise)
                                )
pairwise_distances = torch.tensor(mat['pairwise_distances'][:num_data_points,0]).to(torch.float64)

K1 = torch.tensor(mat['topK']).to(torch.float64)
K2 = torch.tensor(mat['side2K']).to(torch.float64)
K3 = torch.tensor(mat['side1K']).to(torch.float64)

R = torch.eye(3, dtype=torch.float64)
R_stereo_cam_1 = roty(pi/2 - pi) @ rotz(pi/2) @ R
T_stereo_cam_1 = torch.tensor([-530., -0., -250.], dtype=torch.float64)[:, None]
T_stereo_cam_1 = R_stereo_cam_1.T @ T_stereo_cam_1

R_stereo_cam_2 = rotx(pi/2) @ rotz(pi) @ R
T_stereo_cam_2 = torch.tensor([0., -530., -250.], dtype=torch.float64)[:, None]
T_stereo_cam_2 = R_stereo_cam_2.T @ T_stereo_cam_2

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_2 = torch.tensor([K3[0,2] - 1, K3[1,2] - 1], dtype=torch.float64)

focal_length_cam_1 = (K1[0,0] + K1[1,1]) / 2
focal_length_cam_2 = (K2[0,0] + K2[1,1]) / 2
focal_length_cam_3 = (K3[0,0] + K2[1,1]) / 2


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

def freeze_camera_intrinsics(arena):
    arena.camera1.focal_length_pixels.requires_grad = False
    arena.camera1.principal_point_pixel.requires_grad = False
    arena.principal_point_pixel_cam_1.requires_grad = False
    arena.focal_length_cam_1.requires_grad = False

def freeze_stereocamera(arena):
    arena.focal_length_cam_1.requires_grad = False
    arena.focal_length_cam_2.requires_grad = False
    arena.principal_point_pixel_cam_1.requires_grad = False
    arena.principal_point_pixel_cam_2.requires_grad = False
    arena.stereo_camera1_angles.requires_grad = False
    arena.stereo_camera2_angles.requires_grad = False
    arena.stereocam1_r1.requires_grad = False
    arena.stereocam2_r1.requires_grad = False

def freeze_camera_angles(arena):
    arena.stereo_camera1_angles.requires_grad = False
    arena.stereo_camera2_angles.requires_grad = False

def freeze_tank_orientation(arena):
    arena.outer_tank_angles.requires_grad = False
    arena.inner_tank_angles.requires_grad = False

def freeze_tank_thickness(arena):
    arena.outer_tank_thickness.requires_grad = False
    arena.inner_tank_thickness.requires_grad = False

def freeze_tank_size(arena):
    arena.outer_tank_size.requires_grad = False
    arena.inner_tank_size.requires_grad = False

def freeze_tank_position(arena):
    arena.outer_tank_center.requires_grad = False
    arena.inner_tank_center.requires_grad = False

def freeze_refractive_indices(arena):
    arena.refractive_index_acrylic.requires_grad = False
    arena.refractive_index_water.requires_grad = False

def unfreeze_all_parameters(arena):
    for param in arena.parameters():
        param.requires_grad = True


#%% Initialize an Arena instance
refractive_index_acrylic = torch.tensor([1.48], dtype=torch.float64)
refractive_index_water = torch.tensor([1.33], dtype=torch.float64)
outer_tank_axes = torch.tensor([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
], 
dtype=torch.float64
)
inner_tank_axes = torch.tensor([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
], 
dtype=torch.float64
)
plane_temp = Plane(axes=outer_tank_axes)
outer_tank_angles = torch.tensor(
    [plane_temp.alpha, plane_temp.beta, plane_temp.gamma],
    dtype=torch.float64,
)
outer_tank_center = torch.tensor(
    [245., -10., 252.],
    dtype=torch.float64,
) # Center of the side plane in the world frame

outer_tank_size = torch.tensor(
    [508., 530., 152.],
    dtype=torch.float64,
) # 508 mm = 20 inches

outer_tank_thickness = torch.tensor(
    [12.7],
    dtype=torch.float64
)

outer_tank_angles = torch.tensor(
    [plane_temp.alpha, plane_temp.beta, plane_temp.gamma],
    dtype=torch.float64,
)

inner_tank_angles = torch.tensor(
    [plane_temp.alpha, plane_temp.beta, plane_temp.gamma],
    dtype=torch.float64,
)

plane_temp = Plane(axes=inner_tank_axes)
inner_tank_distance = torch.tensor(
    [101., 10., 196.],
    dtype=torch.float64,
) # Center of the side plane in the world frame

inner_tank_size = torch.tensor(
    [250., 270., 152.],
    dtype=torch.float64,
)

inner_tank_thickness = torch.tensor(
    [3.1],
    dtype=torch.float64
)

arena = Arena_Akihiro_fish_tank_pairwise_distances(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            principal_point_pixel_cam_2,
            focal_length_cam_1, 
            focal_length_cam_2,
            focal_length_cam_3,
            R_stereo_cam_1,
            T_stereo_cam_1, 
            R_stereo_cam_2,
            T_stereo_cam_2,
            outer_tank_angles=outer_tank_angles,
            outer_tank_center=outer_tank_center,
            outer_tank_size=outer_tank_size,
            outer_tank_thickness=outer_tank_thickness,
            inner_tank_angles=inner_tank_angles,
            inner_tank_distance=inner_tank_distance,
            inner_tank_size=inner_tank_size,
            inner_tank_thickness=inner_tank_thickness,
            refractive_index_acrylic=refractive_index_acrylic,
            refractive_index_water=refractive_index_water)

#freeze_camera_parameters(arena.camera1)
#freeze_stereocamera(arena)
freeze_tank_orientation(arena)
freeze_tank_thickness(arena)
freeze_refractive_indices(arena)
freeze_camera_angles(arena)
# freeze_camera_intrinsics(arena)
#freeze_stereocamera(arena)
#freeze_tank_size(arena)

#%% Training and validation functions
def train_two_cams(model, train_loader, criterion, plot=False):    
    model.train()
    closest_dist_loss = 0.
    total_loss = 0.
    pairwise_dist_loss = 0.
    intersection_loss = 0.
    bad_rays_penalty = 0.
    
    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches        
        for input, pairwise_distance_batch in train_loader:         
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            recon_3D, closest_distance, pairwise_distance_recon, intersection_penalty, num_bad_rays = model(
                input.T)
            closest_distance_loss = closest_distance.sum()
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum()
            loss = 1e1 * closest_distance_loss + 1e1 * pairwise_distance_loss + 1e4 * intersection_penalty.sum() + 1e3 * num_bad_rays.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            closest_dist_loss += closest_distance_loss.item()
            intersection_loss += intersection_penalty.sum().item()
            bad_rays_penalty += 1e3 * num_bad_rays.sum().item()
            pairwise_dist_loss += pairwise_distance_loss.item()
            
        if plot:
            # 3D plot
            plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.scatter(
                recon_3D[0,:].detach().numpy(),
                recon_3D[1,:].detach().numpy(),
                recon_3D[2,:].detach().numpy(),
                color='g',
                s=5,
                label='Estimate',
            )
                
    return total_loss / len(train_loader.dataset), pairwise_dist_loss / len(train_loader.dataset), intersection_loss / len(train_loader.dataset), closest_dist_loss / len(train_loader.dataset), bad_rays_penalty / len(train_loader.dataset)


def validate(model, val_loader, criterion):
    model.eval()
    closest_dist_loss = 0.
    total_loss = 0.
    pairwise_dist_loss = 0.
    intersection_loss = 0.
    bad_rays_penalty = 0.
    
    with torch.no_grad():
        for input, pairwise_distance_batch in val_loader:
            recon_3D, closest_distance, pairwise_distance_recon, intersection_penalty, num_bad_rays = model(input.T)
            closest_distance_loss = closest_distance.sum()
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum()
            loss = 1e1 * closest_distance_loss + 1e1 * pairwise_distance_loss + 1e3 * intersection_penalty.sum() + 1e3 * num_bad_rays.sum()
            total_loss += loss.item()
            closest_dist_loss += closest_distance_loss.item()
            pairwise_dist_loss += pairwise_distance_loss.item()
            bad_rays_penalty += 1e3 * num_bad_rays.sum().item()
            intersection_loss += intersection_penalty.sum().item()
    return total_loss / len(val_loader.dataset), pairwise_dist_loss / len(val_loader.dataset), intersection_loss / len(val_loader.dataset), closest_dist_loss / len(val_loader.dataset), bad_rays_penalty / len(val_loader.dataset)


#%% Visualize arena initialization
camera2, camera3, side_plane11_outer, side_plane12_outer, side_plane21_outer, side_plane22_outer, side_plane11_inner, side_plane12_inner, side_plane21_inner, side_plane22_inner, top_plane1_outer, top_plane2_outer, top_plane1_inner, top_plane2_inner, visualized_pixels = arena.visualize(pixels_all_cams, rand_sample=True, color_labels=True)
plt.savefig(f'{outputs_dir}/initialized_arena.png')
recon_3D, closest_distance, pairwise_distance_recon, intersection_penalty, num_bad_rays = arena(
                pixels_all_cams)
d_pairwise_distance = (pairwise_distance_recon - pairwise_distances)
pairwise_distance_loss = torch.abs(d_pairwise_distance).mean()
print(f'Pairwise distances at initialization {pairwise_distance_loss}')
""

#%% Training setup
batch_size=1024
rand_ind = torch.randperm(output_cam_0_pairwise.shape[1])
test_dataset_size = 150
pixels_all_cams_test = pixels_all_cams[:, rand_ind[:test_dataset_size]]
pixels_all_cams = pixels_all_cams[:,rand_ind[test_dataset_size:]]
pairwise_distances_test = pairwise_distances[rand_ind[:test_dataset_size]]
pairwise_distances = pairwise_distances[rand_ind[test_dataset_size:]]

dataset = CalibrationDataset(pixels_all_cams, pairwise_distances)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_all_cams_train, pixels_all_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_all_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_all_cams_val, batch_size=batch_size, shuffle=False)

num_epochs = 800
optimizer = optim.Adam(arena.parameters(), lr=1
                       )
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)
pixels_virtual_two_cams = pixels_all_cams.to(device)

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
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-2

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-3
            unfreeze_all_parameters(arena)
            freeze_camera_angles(arena)
            freeze_tank_orientation(arena)

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-3
    
    if epoch == 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
            unfreeze_all_parameters(arena)

    if epoch == 1250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4          
        
    if plot:
        plt.title(f'Epoch {epoch}')
    
    if epoch % 100 == 0:
        plot = True
            
    train_loss, train_pairwise_distance_loss, train_intersection_loss, train_closest_distance_loss, _ = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    plot = False    
    train_loss_array.append(train_loss)
    
    val_loss, val_pairwise_distance_loss, val_intersection_loss, val_closest_distance_loss, bad_rays_penalty = validate(model=arena,
             val_loader=val_loader,
             criterion=criterion,
             )
    
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train_loss: {train_loss}')
        print(f'Validation loss for epoch {epoch}: val_loss: {val_loss}, pairwise_distance_loss: {val_pairwise_distance_loss}, intersection_loss: {val_intersection_loss}, closest_distance_loss: {val_closest_distance_loss}, bad_rays_penalty: {bad_rays_penalty}')
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
        print(f'Found better model with validation loss for epoch {epoch}: val_loss: {val_loss}')
        print(f'Validation loss for epoch {epoch}: val_loss: {val_loss}, pairwise_distance_loss: {val_pairwise_distance_loss}, intersection_loss: {val_intersection_loss}, closest_distance_loss: {val_closest_distance_loss}, bad_rays_penalty: {bad_rays_penalty}')
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/val_pairwise_distance_error', val_pairwise_distance_loss, epoch)
    writer.add_scalar('Loss/val_closest_distance_error', val_closest_distance_loss, epoch)
    writer.add_scalar('Parameter/tank/refractive_index_glass', arena.refractive_index_water, epoch)
    writer.add_scalar('Parameter/tank/refractive_index_acrylic', arena.refractive_index_acrylic, epoch)

    writer.add_scalars('Parameter/inner_tanks_angles', {
        'Angle0':arena.inner_tank_angles[0],
         'Angle1':arena.inner_tank_angles[1],
          'Angle2':arena.inner_tank_angles[2]},
            epoch)
    writer.add_scalars('Parameter/inner_tank_size', {
        'Size0':arena.inner_tank_size[0],
         'Size1':arena.inner_tank_size[1],
          'Size2':arena.inner_tank_size[2]},
            epoch)
    writer.add_scalars('Parameter/inner_tank_distance', {
        'Center0':arena.inner_tank_distance[0],
         'Center1':arena.inner_tank_distance[1],
          'Center2':arena.inner_tank_distance[2]},
            epoch)
    
    writer.add_scalars('Parameter/outer_tanks_angles', {
        'Angle0':arena.outer_tank_angles[0],
         'Angle1':arena.outer_tank_angles[1],
          'Angle2':arena.outer_tank_angles[2]},
            epoch)
    writer.add_scalars('Parameter/outer_tank_size', {
        'Size0':arena.outer_tank_size[0],
         'Size1':arena.outer_tank_size[1],
          'Size2':arena.outer_tank_size[2]},
            epoch)
    writer.add_scalars('Parameter/outer_tank_center', {
        'Center0':arena.outer_tank_center[0],
         'Center1':arena.outer_tank_center[1],
          'Center2':arena.outer_tank_center[2]},
            epoch)
    writer.add_scalar('Parameter/focal_length_pixels_0', arena.camera1.focal_length_pixels, epoch)
    writer.add_scalars('Parameter/camera1_principal_point', {
        'Angle0':arena.camera1.principal_point_pixel[0],
         'Angle1':arena.camera1.principal_point_pixel[1]},
            epoch)
    writer.add_scalars('Parameter/camera1_angles', {
            'Angle0':arena.stereo_camera1_angles[0], 
            'Angle1':arena.stereo_camera1_angles[1],
            'Angle2':arena.stereo_camera1_angles[2]},
            epoch)
    writer.add_scalar('Parameter/camera1_focal_length', arena.camera1.focal_length_pixels, epoch)
    writer.add_scalars('Parameter/StereoCam_T', {
            't_stereo_cam_0': arena.T_stereo_cam1[0],
            't_stereo_cam_1': arena.T_stereo_cam1[1],
            't_stereo_cam_2': arena.T_stereo_cam1[2]},
            epoch)
    gt_train_loss_array.append(train_loss)
    gt_val_loss_array.append(val_loss)
    closest_distance_train_loss_array.append(train_closest_distance_loss)
    closest_distance_val_loss_array.append(val_closest_distance_loss)

# %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

recon_3D_test, closest_distance_loss_test, pairwise_distance_recon_test, _, _ = arena(pixels_all_cams_test)
print(f'Closest distance loss: {closest_distance_loss_test.mean()}')
pairwise_distance_loss = torch.abs(pairwise_distances_test - pairwise_distance_recon_test).mean()
print(f'Pairwise distance loss {pairwise_distance_loss}')

output = "Pairwise distance loss: {pairwise_distance_loss}, Closest distance loss: {closest_distance_loss}"
with open(f"{model_checkpoint_dir}/errors.txt", "w") as f:
    f.write(output)

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
plt.plot(epochs, gt_train_loss_array, color='b', 
         label='Ground truth train loss')
plt.plot(epochs, closest_distance_train_loss_array, 
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
arena.visualize(pixels_all_cams_test, color_labels=True, rand_sample=False)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
