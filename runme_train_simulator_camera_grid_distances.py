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
from arenasEfficient import Arena_two_real_cameras_grid_distance
from utils import euclidean_distance

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
virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_undistorted_pairwise'], dtype=torch.float64, requires_grad=True).T - 1.
virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_undistorted_pairwise'], dtype=torch.float64, requires_grad=True).T - 1.
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_undistorted_pairwise'], dtype=torch.float64).T - 1.
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_undistorted_pairwise'], dtype=torch.float64).T - 1.
target_coordinates = torch.tensor(mat['worldPoints_pairwise'], dtype=torch.float64).T
pairwise_distance = torch.tensor(mat['pairwise_distances'][:,0]).to(torch.float64)
stereoParams = mat['stereoParams_export']
K1 = torch.tensor(stereoParams['CameraParameters1K'][0,0]).to(torch.float64)
K2 = torch.tensor(stereoParams['CameraParameters2K'][0,0]).to(torch.float64)
R = torch.tensor(stereoParams['RotationOfCamera2'][0,0]).to(torch.float64)
T = torch.tensor(stereoParams['TranslationOfCamera2'][0,0]).to(torch.float64).T

#principal_point_pixel_cam_0 = torch.tensor([640., 512.]).to(torch.float64)
#principal_point_pixel_cam_1 = torch.tensor([640., 512.]).to(torch.float64)

principal_point_pixel_cam_0 = torch.tensor([K1[0,2] - 1, K1[1,2] - 1], dtype=torch.float64)
principal_point_pixel_cam_1 = torch.tensor([K2[0,2] - 1, K2[1,2] - 1], dtype=torch.float64)

#focal_length_cam_1 = 5208. 
#focal_length_cam_2 = 5208. 

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
arena = Arena_two_real_cameras_grid_distance(principal_point_pixel_cam_0,
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
    repr_loss = 0.
    dist_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.

    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for i, (input, label_2D, label_3D, pairwise_distance_batch) in enumerate(train_loader):         
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_real_loss, pairwise_distance_recon = model(label_2D.T)
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            pairwise_distance_difference = torch.abs(d_pairwise_distance).mean() # Mean absolute difference
            if epoch == 0 and i == 0:
                print(f'Loss for first iteration: Closest_distance_loss: {closest_distance.mean()}, Pairwise_distance_loss: {torch.abs(d_pairwise_distance).mean()}')            
            cam_0_pixels_gt = torch.hstack((label_2D.T[:2,:], label_2D.T[2:4,:]))
            cam_1_pixels_gt = torch.hstack((label_2D.T[4:6,:], label_2D.T[6:8,:]))
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
                    cam_0_pixels_gt[0,rand_ind].detach().numpy(),
                    cam_0_pixels_gt[1,rand_ind].detach().numpy(),
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
                    cam_1_pixels_gt[0,rand_ind].detach().numpy(),
                    cam_1_pixels_gt[1,rand_ind].detach().numpy(),
                    s=0.5,
                    c='g',
                )
            stacked_label_3D = torch.vstack((label_3D[:,:recon_3D.shape[0]], label_3D[:,recon_3D.shape[0]:]))
            triangulation_loss = euclidean_distance(
                            stacked_label_3D.T, recon_3D
                            ).sum()
            reprojection_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            loss = 1 * recon_real_loss.sum()  + 1 * triangulation_loss + 0 * closest_distance_loss + 0 * pairwise_distance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            repr_loss += reprojection_loss.item()
            tr_loss += triangulation_loss.item()
            dist_loss += closest_distance_loss.item()  
            pairwise_dist_loss += pairwise_distance_loss.item()          
            total_loss += loss.item()
            
    return total_loss / len(train_loader.dataset), repr_loss / len(train_loader.dataset), dist_loss / len(train_loader.dataset), tr_loss / len(train_loader.dataset), pairwise_dist_loss / len(train_loader.dataset)


def validate(model, val_dataloader, criterion):
    model.eval()
    val_repr_loss = 0.
    val_dist_loss = 0.
    tr_loss = 0.
    pairwise_dist_loss = 0.
    total_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for input, label_2D, label_3D, pairwise_distance_batch in val_dataloader:
            num_batches += 1
            recon_3D, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_real_loss, pairwise_distance_recon = model(label_2D.T)
            reprojection_loss = recon_real_loss.sum()
            closest_distance_loss = closest_distance.sum()
            stacked_label_3D = torch.vstack((label_3D[:,:recon_3D.shape[0]], label_3D[:,recon_3D.shape[0]:]))
            triangulation_loss = euclidean_distance(
                            stacked_label_3D.T, recon_3D
                            ).sum()
            d_pairwise_distance = (pairwise_distance_recon - pairwise_distance_batch)
            pairwise_distance_loss = torch.abs(d_pairwise_distance).sum() # Sum of root squared error
            pairwise_distance_difference = torch.abs(d_pairwise_distance).mean() # Mean absolute difference
            loss = 1 * recon_real_loss.sum()  + 0 * triangulation_loss + 0 * closest_distance_loss + 0 * pairwise_distance_loss
            val_repr_loss += reprojection_loss.item()
            val_dist_loss += closest_distance_loss.item()
            tr_loss += triangulation_loss.item()
            pairwise_dist_loss += pairwise_distance_loss.item()            
            total_loss += loss.item()
    return total_loss / len(val_loader.dataset), val_repr_loss / len(val_loader.dataset), val_dist_loss / len(val_loader.dataset), tr_loss / len(val_loader.dataset), pairwise_dist_loss / len(val_loader.dataset)


#%% Visualize arena initialization
arena.visualize(pixels_virtual_two_cams)
recon_3D, closest_distance, _, _, reprojection_loss_init, pairwise_distance_recon = arena(pixels_real_two_cams)
pairwise_distance_loss = torch.abs((pairwise_distance_recon.detach() - pairwise_distance))
stacked_label_3D = torch.hstack((target_coordinates[:recon_3D.shape[0],:], target_coordinates[recon_3D.shape[0]:,:]))
triangulation_loss = euclidean_distance(
                            stacked_label_3D, recon_3D
                            )
print(f'Initialization loss: reprojection_error: {reprojection_loss_init.mean()}, pairwise_distance_loss: {pairwise_distance_loss.mean()}, triangulation_loss: {triangulation_loss.mean()}')
plt.savefig(f'{outputs_dir}/initialized_arena.png')


#%% Training setup
batch_size=1028
rand_ind = torch.randperm(pixels_virtual_two_cams.shape[1])
test_dataset_size = 150
pixels_virtual_two_cams_test = pixels_virtual_two_cams[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]
pixels_real_two_cams_test = pixels_real_two_cams[:, rand_ind[:test_dataset_size]]
pairwise_distance_test = pairwise_distance[rand_ind[:test_dataset_size]]

pixels_virtual_two_cams = pixels_virtual_two_cams[:,rand_ind[test_dataset_size:]]
pixels_real_two_cams = pixels_real_two_cams[:, rand_ind[test_dataset_size:]]
target_coordinates = target_coordinates[:, rand_ind[test_dataset_size:]]
pairwise_distance = pairwise_distance[rand_ind[test_dataset_size:]]

dataset = CalibrationDataset(pixels_virtual_two_cams, pixels_real_two_cams, target_coordinates, pairwise_distance)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 50
optimizer = optim.Adam(arena.parameters(), lr=1e-3
                       )
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)
pixels_virtual_two_cams = pixels_virtual_two_cams.to(device)
max_norm = 1.   

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
    train_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    """
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-2

    if epoch == 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-2
    """
    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 10 == 0:
        print(f'Training loss for epoch {epoch}: train loss: {train_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance_loss: {pairwise_distance_loss}')
        if epoch % 200 == 0:
            plot = True
    gt_train_loss_array.append(repr_loss)
    closest_distance_train_loss_array.append(dist_loss)

    val_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss = validate(model=arena,
             val_dataloader=val_loader,
             criterion=criterion,
             )
    scheduler.step(val_loss)
    if epoch % 10 == 0:
        print(f'Validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance loss: {pairwise_distance_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if repr_loss + dist_loss + triangulation_loss < best_loss:
        best_loss = repr_loss + dist_loss + triangulation_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': repr_loss + dist_loss + triangulation_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')
        print(f'Found better model with validation loss for epoch {epoch}: val loss: {val_loss}, repr_loss: {repr_loss}, triangulation_loss: {triangulation_loss}, closest_distance_loss: {dist_loss}, pairwise_distance loss: {pairwise_distance_loss}')
    gt_val_loss_array.append(repr_loss)
    closest_distance_val_loss_array.append(dist_loss)


# %% Validate the model

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])

#%%
target_coordinates_test_ = torch.hstack((target_coordinates_test[:3,:], target_coordinates_test[3:,:]))
recon_3D_test, closest_dist_test, recon_real_1, recon_real_2, reprojection_error_test, pairwise_distance_test_ = arena(pixels_real_two_cams_test)
pixels_real_two_cam_0_test_ = torch.hstack((pixels_real_two_cams_test[:2,:], pixels_real_two_cams_test[2:4,:]))
pixels_real_two_cam_1_test_ = torch.hstack((pixels_real_two_cams_test[4:6,:], pixels_real_two_cams_test[6:8,:]))
test_loss = euclidean_distance(
                            target_coordinates_test_, recon_3D_test
                            ).mean()

pairwise_distance_loss = torch.abs(pairwise_distance_test_ - pairwise_distance_test).mean()
print(f'Pairwise distance loss: {pairwise_distance.mean()}')
print(f'Triangulation loss: {test_loss.mean()}')
print(f'Closest distance loss: {closest_dist_test.mean()}')

reprojection_error_1 = torch.linalg.norm(
                                    recon_real_1 - pixels_real_two_cam_0_test_, dim=0
                                    )
reprojection_error_2 = torch.linalg.norm(
                                    recon_real_2 - pixels_real_two_cam_1_test_, dim=0
                                    )
print(f'Reprojection error: cam_0 : {reprojection_error_1.mean()}, cam_1 : {reprojection_error_2.mean()}')

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
ax.set_title(f'Reprojection error: {((reprojection_error_1.mean()+reprojection_error_2.mean())/2):.2f} pixels', fontsize=22)
plt.legend(fontsize=22)
plt.savefig(f'{model_checkpoint_dir}/reprojection_error.png')

#%%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    target_coordinates_test_[0,:].detach().numpy(),
    target_coordinates_test_[1,:].detach().numpy(),
    target_coordinates_test_[2,:].detach().numpy(),
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
arena.visualize(pixels_real_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
arena_copy = Arena_two_real_cameras_grid_distance(principal_point_pixel_cam_0,
            principal_point_pixel_cam_1, 
            focal_length_cam_1, 
            focal_length_cam_2,
            R,
            T, 
            prism_angles=prism_angles,
            prism_center=prism1_center)
arena_copy.load_state_dict(arena.state_dict())

loss_array_ = []
repr_loss_array_ = []
dist_loss_array_ = [] 
triangulation_loss_array_ = [] 
pairwise_distance_loss_array_ = []

perturbations = torch.linspace(-100, 100, 25)
for pert in perturbations:
    arena_copy.camera1.focal_length_pixels = nn.Parameter(arena.camera1.focal_length_pixels.clone() + pert)
    val_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss = validate(model=arena_copy,
                val_dataloader=val_loader,
                criterion=criterion,
                )
    loss_array_.append(val_loss)
    repr_loss_array_.append(repr_loss)
    triangulation_loss_array_.append(triangulation_loss)
    pairwise_distance_loss_array_.append(pairwise_distance_loss*1e2)

plt.figure(figsize=(15,15))
plt.plot(perturbations, loss_array_, label='Total loss')
plt.plot(perturbations, repr_loss_array_, label='Reprojection loss')
plt.plot(perturbations, triangulation_loss_array_, label='Triangulation loss')
plt.plot(perturbations, pairwise_distance_loss_array_, label='Pairwise distance loss')
plt.legend(fontsize=22)
plt.xlabel('Perturbation', fontsize=22)



# %% Surface plot for reprojection loss and triangulation loss
perturbations = torch.linspace(-100, 100, 15)
perturbations_ = torch.linspace(-100, 100, 15)
loss_array_ = torch.zeros((perturbations.shape[0], perturbations_.shape[0]))
repr_loss_array_ = torch.zeros((perturbations.shape[0], perturbations_.shape[0]))
dist_loss_array_ = torch.zeros((perturbations.shape[0], perturbations_.shape[0]))
triangulation_loss_array_ = torch.zeros((perturbations.shape[0], perturbations_.shape[0]))
pairwise_distance_loss_array_ = torch.zeros((perturbations.shape[0], perturbations_.shape[0]))

for i, pert in enumerate(perturbations):
    for j, pert_ in enumerate(perturbations_):
        arena_copy.camera1.focal_length_pixels = nn.Parameter(arena.camera1.focal_length_pixels.clone() + pert)
        arena_copy.camera1.principal_point_pixel = nn.Parameter(arena.camera1.principal_point_pixel.clone() + pert_)
        val_loss, repr_loss, dist_loss, triangulation_loss, pairwise_distance_loss = validate(model=arena_copy,
                    val_dataloader=val_loader,
                    criterion=criterion,
                    )
        loss_array_[i,j] = val_loss
        repr_loss_array_[i,j] = repr_loss
        triangulation_loss_array_[i,j] = triangulation_loss
        pairwise_distance_loss_array_[i,j] = pairwise_distance_loss * 1e2
    
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(perturbations, perturbations_)
ax.plot_surface(X, Y, loss_array_.detach().numpy(), label='Total loss', color='b')
ax.plot_surface(X, Y, repr_loss_array_.detach().numpy(), label='Reprojection loss', color='r')
ax.plot_surface(X, Y, triangulation_loss_array_.detach().numpy(), label='Triangulation loss', color='g')
ax.plot_surface(X, Y, pairwise_distance_loss_array_.detach().numpy(), label='Pairwise distance loss', color='m')
ax.set_xlabel('Perturbation 1', fontsize=22)
ax.set_ylabel('Perturbation 2', fontsize=22)
ax.set_zlabel('Loss', fontsize=22)
ax.legend(fontsize=22)

# %%

