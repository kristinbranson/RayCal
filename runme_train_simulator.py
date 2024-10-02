# %% Imports
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, visualize_camera_configuration, closest_point, rotx
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
pi = torch.tensor(np.pi)
torch.autograd.set_detect_anomaly(True)

#%%
class CalibrationDataset(Dataset):
    def __init__(self, data, labels):
        # Example data
        self.data = data.T
        self.labels = labels.T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#%% Load camera calibration results
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
mat = sio.loadmat(calibration_results_path)
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_undistorted'], dtype=torch.float32, requires_grad=True).T 
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_undistorted'], dtype=torch.float32, requires_grad=True).T 
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T

principal_point_pixel_cam_0 = [638.040, 492.499] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778, 521.5078]

R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None]
focal_length_cam_1 = 5696.3
focal_length_cam_2 = 5790.3


#%% 
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
prism_corners = sio.loadmat(prism_corners_path)['worldPoints']
prism3_axes = torch.zeros(3,3)
#prism3_axes[:,1] = torch.stack(
#    (torch.tensor(prism_corners[1,:] - prism_corners[0,:]),
#     torch.tensor(prism_corners[2,:] - prism_corners[3,:])
#    )
#).mean(dim=0)
prism3_axes[:,1] = torch.stack(
    (torch.tensor(prism_corners[1,:] - prism_corners[0,:]),
    )
).mean(dim=0)
prism3_axes[:,2] = torch.stack(
    (torch.tensor(prism_corners[3,:] - prism_corners[0,:]),
     torch.tensor(prism_corners[2,:] - prism_corners[1,:])
     )
).mean(dim=0)
prism3_axes[:,0] = torch.linalg.cross(prism3_axes[:,1], prism3_axes[:,2])
prism3_axes = prism3_axes / torch.linalg.norm(prism3_axes, dim=0)
prism1_axes = torch.mm(rotx(pi/2), prism3_axes)

prism_a = 20.
prism_b = 20.
prism3_center = torch.tensor(prism_corners).mean(dim=0)
prism1_center = prism3_center + prism_b/2 * prism1_axes[:,0] - prism_b/2 * prism1_axes[:,2] 
plane = Plane(axes=prism1_axes)
prism_angles = torch.tensor([plane.alpha, plane.beta, plane.gamma])


#%% Define Arena class
class Arena(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena, self).__init__()
        # Camera initialization        
        
        self.camera1 = Camera(
            principal_point_pixel=principal_point_pixel_cam_0, 
            focal_length_pixels=focal_length_cam_0)
        self.camera2 = Camera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1)
        self.camera2.update_camera_pose(R, T)
        
        # Prism initialization
        prism_angles = nn.Parameter(torch.tensor(prism_angles), requires_grad=True)

        refractive_index_glass = torch.tensor(1.5, dtype=torch.float32)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=False)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)

        self.prism = Prism(prism_size=[20.,20.,20.], 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        freeze_camera_parameters(self.camera1)
        freeze_camera_parameters(self.camera2)
        #freeze_individual_planes(self.prism)

    
    def forward(self, pixels_two_cams):
        undistorted_real_pixels_cam_0 = pixels_two_cams[:2, :]
        undistorted_real_pixels_cam_1 = pixels_two_cams[2:, :]
        #cam_1_ray = self.camera1.initialize_ray(undistorted_real_pixels_cam_0)
        cam_1_ray = self.camera1(undistorted_real_pixels_cam_0)
        #cam_2_ray = self.camera2.initialize_ray(undistorted_real_pixels_cam_1)
        cam_2_ray = self.camera2(undistorted_real_pixels_cam_1)
        prism_ray11, prism_ray12, emergent_ray_1 = self.prism(cam_1_ray)
        prism_ray21, prism_ray22, emergent_ray_2 = self.prism(cam_2_ray)
        recon_3D, closest_distance = closest_point(emergent_ray_1, emergent_ray_2)
        return recon_3D, closest_distance, prism_ray11, prism_ray12, prism_ray21, prism_ray22
    
    """
    def forward(self, pixels):
        undistorted_real_pixels_cam_0 = pixels_two_cams[:2, :]
        cam_0_ray = self.camera1(undistorted_real_pixels_cam_0)
        _, _, emergent_ray_0 = self.prism(cam_0_ray)
        return emergent_ray_0
    """

    def visualize(self, pixels_two_cams):
        num_samples = 10
        undistorted_real_pixels_cam_0 = pixels_two_cams[:2, :].clone()
        undistorted_real_pixels_cam_1 = pixels_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_real_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_real_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax)
        prism_ray11, prism_ray12, emergent_ray1 = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax)

        ray = self.camera2.initialize_ray(undistorted_real_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax)
        prism_ray21, prism_ray22, emergent_ray2 = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray2.t *= 25
        emergent_ray2.visualize(fig, ax)
        ax.set_aspect('equal', adjustable='datalim')   


#%% Initialize an Arena instance
#prism_angles = [0., 0., 0.]
prism_distance = torch.tensor(130.)
arena = Arena(principal_point_pixel_cam_0, 
principal_point_pixel_cam_1, 
focal_length_cam_1, 
focal_length_cam_2,
R,
T, 
prism_angles=prism_angles,
prism_center=prism1_center)


# %% Set up the optimizer and criterion
optimizer = optim.Adam(arena.parameters(), lr=1e-1)
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)

def train_two_cams(model, pixels_two_cams_minibatch, target_coordinates_minibatch):
    model.train()
    optimizer.zero_grad()
    output, closest_distance, _, _, _, _ = model(pixels_two_cams_minibatch)
    ground_truth_loss = criterion(output, target_coordinates_minibatch) 
    closest_distance_loss = closest_distance.mean()
    loss = 0 * closest_distance_loss + ground_truth_loss
    loss.backward()
    optimizer.step()
    return ground_truth_loss.item(), closest_distance_loss.item()

def train_one_cam(model, pixels):
    model.train()
    optimizer.zero_grad()
    output_ray = model(pixels)
    closest_distance_loss = output_ray.distance_to_point(target_coordinates).mean()
    closest_distance_loss.backward()
    optimizer.step()
    return closest_distance_loss.item()

#pixels_two_cams = [undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1]
pixels_two_cams = torch.vstack((undistorted_real_pixels_cam_0, undistorted_real_pixels_cam_1)).to(device)
pixels = undistorted_real_pixels_cam_0

#%%
arena.visualize(pixels_two_cams)
plt.savefig('outputs/initialized_arena.png')

#%% Dataset
batch_size=256

dataset = CalibrationDataset(pixels_two_cams, target_coordinates)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation

# Split the dataset
pixels_two_cams_train, pixels_two_cams_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_two_cams_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_two_cams_val, batch_size=batch_size, shuffle=False)

#%%
#train_two_cams(arena, pixels_two_cams_train)


# %%
training_losses = []
for epoch in tqdm(range(1000)):
    with torch.autograd.set_detect_anomaly(True):
        for input, label in train_loader:
            gt_loss, dist_loss = train_two_cams(arena, input.T, label.T)
        #dist_loss = train_one_cam(arena, pixels_two_cams)
        if epoch % 100 == 0:
            #print(f'Epoch: {epoch},  Distance loss: {dist_loss}')
            print(f'Epoch: {epoch},  gt_loss : {gt_loss},  dist_loss: {dist_loss}')
    #training_losses.append(dist_loss.detach().numpy())
    training_losses.append((gt_loss + dist_loss))

#%%
plt.figure()
plt.plot(training_losses)
plt.savefig('outputs/training_loss.png')

# %%
arena.visualize(pixels_two_cams)
plt.savefig('outputs/final_arena.png')
# %%
