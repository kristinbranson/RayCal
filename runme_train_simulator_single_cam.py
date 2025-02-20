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
focal_length_cam_1 = 5696.3 # in pixels
focal_length_cam_2 = 5790.3 # in pixels


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


#%% Initialize an arena instance
prism_distance = torch.tensor(130.) # Not used if you're using fiduciary markers for initialization
arena = Arena_single_camera(principal_point_pixel_cam_0, 
focal_length_cam_1, 
prism_angles=prism_angles,
prism_center=prism1_center)
pixels = torch.tensor(undistorted_real_pixels_cam_0)


#%% Training and validation functions
def train_one_cam(model, train_loader, criterion):
    model.train()
    dist_loss = 0.
    for input, label in train_loader:
        output_ray = model(input.T)
        closest_distance_loss = output_ray.distance_to_point(label.T).sum()
        closest_distance_loss.backward()
        optimizer.step()
        dist_loss += closest_distance_loss.item()
        return dist_loss / len(train_loader.dataset)

def validate_one_cam(model, val_loader, criterion):
    model.eval()
    dist_loss = 0.
    with torch.no_grad():
        for input, label in val_loader:
            output_ray = model(input.T)
            closest_distance_loss = output_ray.distance_to_point(label.T).sum()
            dist_loss += closest_distance_loss.item()
        return dist_loss / len(train_loader.dataset)



#%% Visualize arena initialization
arena.visualize(pixels)
plt.savefig('outputs/initialized_arena.png')


#%% Training setup
batch_size=512
rand_ind = torch.randperm(pixels.shape[1])
test_dataset_size = 150
pixels_test = pixels[:, rand_ind[:test_dataset_size]]
target_coordinates_test = target_coordinates[:, rand_ind[:test_dataset_size]]

pixels = pixels[:,test_dataset_size:]
target_coordinates = target_coordinates[:, test_dataset_size:]

dataset = CalibrationDataset(pixels, target_coordinates)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size   # Remaining 20% for validation
pixels_train, pixels_val = random_split(
    dataset, [train_size, val_size]
    )
train_loader = DataLoader(pixels_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(pixels_val, batch_size=batch_size, shuffle=False)

num_epochs = 50

optimizer = optim.Adam(arena.parameters(), lr=1e-2)

criterion = torch.nn.MSELoss()


device = torch.device("cpu")

arena.to(device)

pixels = pixels.to(device)


# %% Training loop
gt_train_loss_array = []
closest_distance_train_loss_array = []
gt_val_loss_array = []
closest_distance_val_loss_array = []

for epoch in tqdm(range(num_epochs)):
    dist_loss = train_one_cam(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    )
    if epoch % 5 == 0:
        print(f'Training loss for epoch {epoch}: dist_loss: {dist_loss}')
    closest_distance_train_loss_array.append(dist_loss)

    dist_loss = validate_one_cam(model=arena,
             val_loader=val_loader,
             criterion=criterion,
             )
    if epoch % 5 == 0:
        print(f'Validation loss for epoch {epoch}: dist_loss: {dist_loss}')
    closest_distance_val_loss_array.append(dist_loss)


#%% Make plots after training
plt.figure()
plt.plot(closest_distance_train_loss_array, 
         color='r', 
         label='Closest distance train loss')
plt.plot(closest_distance_val_loss_array, 
         color='r', 
         label='Closest distance val loss',
         linestyle='--')

#%% 
# %% Validate the model
ray = arena(pixels_test)
closest_distance_loss = ray.distance_to_point(target_coordinates_test).mean()         
print(f'Test loss: {closest_distance_loss}')

#%% Test loss
plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.savefig('outputs/training_loss.png')


# Visualize trained arena
plt.figure()
arena.visualize(pixels)
plt.savefig('outputs/final_arena.png')
# %%