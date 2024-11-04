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
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)
now = datetime.datetime.now()
model_checkpoint_dir = f'{outputs_dir}/model_checkpoints/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
if load_checkpoint:
    model_checkpoint_dir = ''
os.makedirs(model_checkpoint_dir, exist_ok=True)

mat = sio.loadmat(calibration_results_path)
virtual_pixels_cam_0 = torch.tensor(mat['output_data_cam_02_undistorted'], dtype=torch.float64, requires_grad=True).T 
virtual_pixels_cam_1 = torch.tensor(mat['output_data_cam_13_undistorted'], dtype=torch.float64, requires_grad=True).T 
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_undistorted'], dtype=torch.float64).T 
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_undistorted'], dtype=torch.float64).T 
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float64).T

principal_point_pixel_cam_0 = [638.040, 492.499] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778, 521.5078] # This comes from the calibration results


R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T.to(torch.float64)
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None].to(torch.float64)
focal_length_cam_1 = 5696.3 # in pixels
focal_length_cam_2 = 5709.3 # in pixels


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


#%% Define Arena class
class Arena(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = Camera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                             dtype=torch.float64),
                                                requires_grad=True
                                                )


        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.5, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float64), requires_grad=True)
        #prism_size = torch.tensor([20.,20.,20.], dtype=torch.float64)
        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        
        #freeze_camera_parameters(self.camera1)
        #freeze_camera_parameters(self.camera2)
        #freeze_individual_planes(self.prism)
    

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1):
        camera2 = Camera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2


    def forward(self, pixels_virtual_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)
        distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
        distorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:,:]
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_MLP(distorted_virtual_pixels_cam_0)
        undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels_MLP(distorted_virtual_pixels_cam_1)
        distorted_virtual_pixels_cam_0_verify = self.camera1.distort_pixels_MLP(undistorted_virtual_pixels_cam_0)
        distorted_virtual_pixels_cam_1_verify = camera2.distort_pixels_MLP(undistorted_virtual_pixels_cam_1)
        #undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :]
        #undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :]
        cam_1_ray = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray = camera2(undistorted_virtual_pixels_cam_1)
        prism_ray11, prism_ray12, emergent_ray_1, intersection_penalty_1 = self.prism(cam_1_ray)
        prism_ray21, prism_ray22, emergent_ray_2, intersection_penalty_2 = self.prism(cam_2_ray)
        recon_3D, closest_distance = closest_point(emergent_ray_1, emergent_ray_2)
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R2 = R_stereo_cam
        T2 = self.T_stereo_cam
        recon_undistorted_virtual_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_undistorted_virtual_pixels_cam_1 = camera2.reproject(recon_3D, R2, R2 @ T2)
        recon_distorted_virtual_pixels_cam_0 = self.camera1.distort_pixels_MLP(recon_undistorted_virtual_pixels_cam_0)
        recon_distorted_virtual_pixels_cam_1 = camera2.distort_pixels_MLP(recon_undistorted_virtual_pixels_cam_1)
        #recon_distorted_virtual_pixels_cam_0 = recon_undistorted_virtual_pixels_cam_0
        #recon_distorted_virtual_pixels_cam_1 = recon_undistorted_virtual_pixels_cam_1
        return recon_3D, closest_distance, recon_distorted_virtual_pixels_cam_0, recon_distorted_virtual_pixels_cam_1, distorted_virtual_pixels_cam_0_verify, distorted_virtual_pixels_cam_1_verify, intersection_penalty_1, intersection_penalty_2


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, emergent_ray1, intersection_penalty_1 = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
        )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray21, prism_ray22, emergent_ray2, intersection_penalty_2 = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = prism_ray21.visualize(fig, ax, color_labels)
        fig, ax = prism_ray22.visualize(fig, ax, color_labels)
        emergent_ray2.t *= 25
        emergent_ray2.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   


#%% Initialize an Arena instance
prism_distance = torch.tensor(130.) # Not used if you're using fiduciary markers for initialization
arena = Arena(principal_point_pixel_cam_0,
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
    
    with torch.autograd.set_detect_anomaly(True):
        # Iterate over minibatches
        if plot:
            plt.figure()
        for input, label_2D, label_3D in train_loader:         
            num_examples = input.shape[0]       
            optimizer.zero_grad()
            output, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_dist_1_verify, recon_dist_2_verify, int_pen_1, int_pen_2 = model(
                input.T)
            
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
            #ground_truth_loss = criterion(output, label.T)
            recon_pixels = torch.cat((recon_distorted_pixels_1,
            recon_distorted_pixels_2), dim=0)
            recon_pixels_verify = torch.cat((recon_dist_1_verify,
                                            recon_dist_2_verify), dim=0)
            distortion_loss = torch.norm(
                    recon_pixels_verify.T - input, 
                    p=2, 
                    dim=1).sum()
            
            ground_truth_loss = torch.norm(
                    recon_pixels.T - label_2D, 
                    p=2, 
                    dim=1).sum()
            
            closest_distance_loss = closest_distance.sum()
            loss = ground_truth_loss + 25 * closest_distance_loss + 10 * distortion_loss
            loss.backward()
            optimizer.step()
            gt_loss += ground_truth_loss.item()
            dist_loss += closest_distance_loss.item()
            
    return gt_loss / len(train_loader.dataset), dist_loss / len(train_loader.dataset)


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
    recon_3D = []
    input_pixels = []
    labels = []
    val_gt_loss = 0.
    val_dist_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for input, label_2D, label_3D in val_dataloader:
            output, closest_distance, recon_distorted_pixels_1, recon_distorted_pixels_2, recon_dist_1_verify, recon_dist_2_verify = model(input.T)
            ground_truth_loss = criterion(output, label_3D.T)
            recon_pixels = torch.cat((recon_distorted_pixels_1,
            recon_distorted_pixels_2), dim=0)
            recon_pixels_verify = torch.cat((recon_dist_1_verify,
                                            recon_dist_2_verify), dim=0)
            distortion_loss = torch.norm(
                    recon_pixels_verify.T - input, 
                    p=2, 
                    dim=1).sum()
            ground_truth_loss = torch.norm(
                    recon_pixels.T - label_2D, 
                    p=2, 
                    dim=1).sum()
            closest_distance_loss = closest_distance.sum()
            recon_3D.append(output)
            input_pixels.append(input)
            labels.append(label_3D)
            val_gt_loss += ground_truth_loss.item()
            val_dist_loss += closest_distance_loss.item()
            num_batches += 1
    return val_gt_loss / len(val_dataloader.dataset), val_dist_loss / len(val_dataloader.dataset)


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

num_epochs = 650
optimizer = optim.Adam(arena.parameters(), lr=5e-4
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

plot = False
for epoch in tqdm(range(num_epochs)):
    gt_loss, dist_loss = train_two_cams(model=arena, 
                    train_loader=train_loader, 
                    criterion=criterion,
                    plot=plot
                    )
    if plot:
        plt.title(f'Epoch {epoch}')
    plot = False
    if epoch % 5 == 0:
        print(f'Training loss for epoch {epoch}: gt_loss: {gt_loss}, dist_loss: {dist_loss}')
        if epoch % 100 == 0:
            plot = True
    gt_train_loss_array.append(gt_loss)
    closest_distance_train_loss_array.append(dist_loss)

    gt_loss, dist_loss = validate(model=arena,
             val_dataloader=val_loader,
             criterion=criterion,
             )
    if epoch % 5 == 0:
        print(f'Validation loss for epoch {epoch}: gt_loss: {gt_loss}, dist_loss: {dist_loss}')
        # Save checkpoint
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': gt_loss + dist_loss,
                }, f'{model_checkpoint_dir}/checkpoint_{epoch}.pth')
    if gt_loss + dist_loss < best_loss:
        best_loss = gt_loss + dist_loss
        torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': arena.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': gt_loss + dist_loss,
                }, f'{model_checkpoint_dir}/best_checkpoint.pth')

    gt_val_loss_array.append(gt_loss)
    closest_distance_val_loss_array.append(dist_loss)


# %% Validate the model
PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

recon_3D_test, closest_dist_test, _, _, _, _  = arena(pixels_virtual_two_cams_test)
test_loss = torch.norm(recon_3D_test - target_coordinates_test, 
        p=2, 
        dim=0).mean()


print(f'Test loss: {test_loss}')
cameraMatrix1 = torch.tensor([[5696.3, 0., 638.040, 0.], [0, 5696.3 , 492.499, 0.], [0., 0., 1., 0.]])
homogeneous_recon_3D_test = torch.vstack((recon_3D_test,
                                          torch.zeros(1,recon_3D_test.shape[1])))
homogeneous_target_coordinates_test = torch.vstack((target_coordinates_test,
                                          torch.zeros(1,target_coordinates_test.shape[1])))
reprojection_pixels_test_homogeneous = cameraMatrix1 @ homogeneous_recon_3D_test
reprojection_pixels_test = reprojection_pixels_test_homogeneous[:2,:] / reprojection_pixels_test_homogeneous[-1,:][None,:]
reprojection_error = torch.linalg.norm(
                                    reprojection_pixels_test - pixels_real_two_cams_test[:2,:], dim=0
                                    )
print(f'Reprojection error: {reprojection_error.mean()}')

plt.figure()
plt.scatter(
    reprojection_pixels_test[0,:].detach().numpy(),
    reprojection_pixels_test[1,:].detach().numpy(),
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
ax.set_xlabel('X (pixels)', fontsize=16)
ax.set_ylabel('Y (pixels)', fontsize=16)
ax.set_title(f'Reprojection error: {reprojection_error.mean():.2f} pixels', fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f'{model_checkpoint_dir}/reprojection_error.png')



#%% Test loss
#%% Make plots after training
plt.figure()
plt.plot(gt_train_loss_array, color='b', 
         label='Ground truth train loss')
plt.plot(closest_distance_train_loss_array, 
         color='r', 
         label='Closest distance train loss')
plt.plot(gt_val_loss_array, color='b', 
         label='Ground truth val loss',
         linestyle='--')
plt.plot(closest_distance_val_loss_array, 
         color='r', 
         label='Closest distance val loss',
         linestyle='--')
plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss (mm)', fontsize=16)
plt.savefig(f'{model_checkpoint_dir}/training_loss.png')


#%% Visualize trained arena
arena.visualize(pixels_virtual_two_cams_test, color_labels=True)
plt.savefig(f'{model_checkpoint_dir}/final_arena.png')
# %%
