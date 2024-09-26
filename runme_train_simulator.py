# %% Imports
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, visualize_camera_configuration, closest_point
import matplotlib.pyplot as plt
import numpy as np  
import torch
import scipy.io as sio
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
#from torchsummary import summary
pi = torch.tensor(np.pi)
torch.autograd.set_detect_anomaly(True)

#%% Load camera calibration results
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
mat = sio.loadmat(calibration_results_path)
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_undistorted'], dtype=torch.float32, requires_grad=True).T - 1
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_undistorted'], dtype=torch.float32, requires_grad=True).T - 1
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T

principal_point_pixel_cam_0 = [638.040 - 1, 492.499 - 1] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778 - 1, 521.5078 - 1]

R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None]
focal_length_cam_1 = 19.65
focal_length_cam_2 = 19.69

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



#%% Define Arena class
class Arena(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance,
    prism_angles):
        super(Arena, self).__init__()
        # Camera initialization        
        camera1_axes = torch.tensor([
            [0., 0., -1.],
            [0., 1., 0.],
            [1., 0., 0.]
        ])
        self.camera1 = Camera(
            principal_point_pixel=principal_point_pixel_cam_0, 
            focal_length=focal_length_cam_0, 
            axes=camera1_axes)
        self.camera2 = Camera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length=focal_length_cam_1,
            axes=camera1_axes)
        self.camera2.update_camera_pose(R, T)
        
        # Prism initialization
        prism_angles = nn.Parameter(torch.tensor(prism_angles), requires_grad=True)

        refractive_index_glass = torch.tensor(1.5, dtype=torch.float32)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=False)
        prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
        prism_center[0] = -15.
        prism_center = nn.Parameter(prism_center, requires_grad=True)
        self.prism = Prism(prism_size=[30.,30.,30.], 
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
prism_angles = [0., 0., 0.]
prism_distance = torch.tensor(130.)
arena = Arena(principal_point_pixel_cam_0, 
principal_point_pixel_cam_1, 
focal_length_cam_1, 
focal_length_cam_2, 
R, 
T, 
prism_distance,
prism_angles)


# %% Set up the optimizer and criterion
optimizer = optim.Adam(arena.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
device = torch.device("cpu")
arena.to(device)

def train_two_cams(model, pixels_two_cams):
    model.train()
    optimizer.zero_grad()
    output, closest_distance, _, _, _, _ = model(pixels_two_cams)
    ground_truth_loss = criterion(output, target_coordinates) 
    closest_distance_loss = closest_distance.mean()
    loss = closest_distance_loss + ground_truth_loss
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

#%%
train_two_cams(arena, pixels_two_cams)
#train_one_cam(arena, pixels)


# %%
training_losses = []
for epoch in tqdm(range(1000)):
    with torch.autograd.set_detect_anomaly(True):
        gt_loss, dist_loss = train_two_cams(arena, pixels_two_cams)
        #dist_loss = train_one_cam(arena, pixels_two_cams)
    if epoch % 100 == 0:
        #print(f'Epoch: {epoch},  Distance loss: {dist_loss}')
        print(f'Epoch: {epoch},  gt_loss : {gt_loss},  dist_loss: {dist_loss}')
    #training_losses.append(dist_loss.detach().numpy())
    training_losses.append((gt_loss + dist_loss))

#%%
plt.plot(training_losses)
# %%
arena.visualize(pixels_two_cams)
# %%
