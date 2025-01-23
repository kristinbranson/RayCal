import torch
from arenasEfficient import Arena_fish_tank_pairwise_distances
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

annotations_file_path = f'/groups/branson/bransonlab/aniket/camera_alignment/annotations.pkl'
annotations_file_path_side = f'/groups/branson/bransonlab/aniket/camera_alignment/calibration_data_initializations/sam2/cam_0_jpg_pi_754-880_mask.npz'
annotations_file_path_top = f'/groups/branson/bransonlab/aniket/camera_alignment/calibration_data_initializations/sam2/cam_1_jpg_pi_754-880_mask.npz'
im_files_path_side = f'/groups/branson/bransonlab/aniket/camera_alignment/calibration_data_initializations/fish_videos/cam_0'
im_files_path_top = f'/groups/branson/bransonlab/aniket/camera_alignment/calibration_data_initializations/fish_videos/cam_1'
output_video_path = f'/groups/branson/bransonlab/aniket/camera_alignment/calibration_data_initializations/fish_videos/reconstruction_video'
save_video = True
frame_rate = 30

num_frames = len(os.listdir(im_files_path_side))
try:
    len(os.listdir(im_files_path_top)) == num_frames
except:
    raise ValueError('Number of frames in top and side camera videos do not match')


im_files_side = [os.path.join(im_files_path_side, f'image_{frame_id}.png') for frame_id in range(1, num_frames)]
im_files_top = [os.path.join(im_files_path_top, f'image_{frame_id}.png') for frame_id in range(1, num_frames)]


def draw_axes(image, center_x, center_y, length, color, axis1_name, axis2_name):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = length / 75
    font_thickness = 2
    # Draw the y-axis
    cv.line(image, (center_x, center_y), (center_x + length, center_y), color, 2)
    # Draw the x-axis
    cv.line(image, (center_x, center_y), (center_x, center_y - length), color, 2)

    # Add arrowheads to the axes
    arrow_length = length // 5
    cv.arrowedLine(image, (center_x + length - arrow_length, center_y), (center_x + length, center_y), color, 2, tipLength=0.5)
    cv.arrowedLine(image, (center_x, center_y - length + arrow_length), (center_x, center_y - length), color, 2, tipLength=0.5)
    cv.putText(image, axis1_name, (center_x + length + length // 8, center_y + length // 6), font, font_scale, color, font_thickness, cv.LINE_AA)

    # Label for Y-axis
    cv.putText(image, axis2_name, (center_x - length // 2 - (length // 4) * (len(axis2_name) - 1), center_y - length + length // 4), font, font_scale, color, font_thickness, cv.LINE_AA)
    return image


def get_plot_limits(recon_3D, zoom_factor=1):
    """
    Get the limits for the 3D plot of 3-D reconstructed trajectory
    recon_3D: torch.Tensor of shape (3, num_annotations, num_keypoints)
    zoom_factor: (float) Factor to zoom out the plot
    """
    zoom_out_factor = 1
    x_max = recon_3D[0, ...].max()
    x_min = recon_3D[0, ...].min()
    x_width = x_max - x_min
    x_min = x_min - zoom_out_factor * (x_width)
    x_max = x_max + zoom_out_factor * (x_width)

    y_max = recon_3D[1, ...].max()
    y_min = recon_3D[1, ...].min()
    y_width = y_max - y_min
    y_min = y_min - zoom_out_factor * (y_width)
    y_max = y_max + zoom_out_factor * (y_width)

    z_max = recon_3D[2, ...].max()
    z_min = recon_3D[2, ...].min()
    z_width = z_max - z_min
    z_min = z_min - zoom_out_factor * (z_width)
    z_max = z_max + zoom_out_factor * (z_width)
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_annotations_from_pkl(annotations_file_path):
    with open(annotations_file_path, 'rb') as f:
        ann = pickle.load(f)
    C_side = torch.tensor(ann['side']).to(torch.float64)
    C_top = torch.tensor(ann['top']).to(torch.float64)
    if C_side.shape[0] != 2:
        C_side = C_side.T
    if C_top.shape[0] != 2:
        C_top = C_top.T
    try:
        assert C_side.shape[1] == C_top.shape[1]
    except:
        raise ValueError('Number of annotations in side and top camera do not match')

    try:
        assert C_side.shape[0] == 2
    except:
        raise ValueError('Annotations in side camera are not of shape (2, num_annotations)')
    return C_side, C_top


def get_annotations_from_sam_npz(annotations_file_path, cam='side'):
    ann = np.load(annotations_file_path)
    num_frames = sum(1 for key in ann if key.startswith('frame_'))
    C_tensor = torch.zeros(2, num_frames).to(torch.float64)
    for frame_id in range(num_frames):
        seg_frame = ann[f'frame_{frame_id}']
        if cam == 'side':
            seg_frame = seg_frame[::-1, ::-1]
        mask_xy = np.argwhere(seg_frame == 1)
        cent = torch.tensor(mask_xy.mean(axis=0))
        C_tensor[:, frame_id] = torch.flip(cent, [0])
    return C_tensor


fig = plt.figure(figsize=(40, 18))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])  # Equal width for both columns
ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Larger subplot spanning both columns
ax2 = fig.add_subplot(gs[0, 1])  # Smaller subplot on the left
C_side = get_annotations_from_sam_npz(annotations_file_path_side, 'side')
C_top = get_annotations_from_sam_npz(annotations_file_path_top, 'top')
#C_side, C_top = get_annotations_from_pkl(annotations_file_path)

def update(frame_id):
    ax1.cla()
    pad = 50 # padding between plot of top camera image and side camera image
    ax1.plot(recon_3D[0, :frame_id+1],
            recon_3D[1, :frame_id+1],
            recon_3D[2, :frame_id+1],   
            color='tab:blue')

    ax1.scatter(recon_3D[0, frame_id],
            recon_3D[1, frame_id],
            recon_3D[2, frame_id],   
            color='tab:red',
            s=40)
    
    x_tick_step = 20
    y_tick_step = 20
    z_tick_step = 10
    x_min_ = x_min // x_tick_step * x_tick_step
    x_max_ = (1 + x_max // x_tick_step) * x_tick_step
    y_min_ = y_min // y_tick_step * y_tick_step
    y_max_ = (1 + y_max // y_tick_step) * y_tick_step
    z_min_ = z_min // z_tick_step * z_tick_step
    z_max_ = (1 + z_max // z_tick_step) * z_tick_step
    ax1.set_xlim(x_min_, x_max_)
    ax1.set_ylim(y_min_, y_max_)
    ax1.set_zlim(z_min_, z_max_)

    # Create new ticks based on the limits and step
    x_ticks = np.arange(np.floor(x_min_), np.ceil(x_max_) + x_tick_step, x_tick_step)
    y_ticks = np.arange(np.floor(y_min_), np.ceil(y_max_) + y_tick_step, y_tick_step)
    z_ticks = np.arange(np.floor(z_min_), np.ceil(z_max_) + z_tick_step, z_tick_step)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_zticks(z_ticks)
    ax1.set_xticklabels(x_ticks, fontsize=18)
    ax1.set_yticklabels(y_ticks, fontsize=18)
    ax1.set_zticklabels(z_ticks, fontsize=18)
    ax1.tick_params(axis='x', pad=10)  # Increase padding for x-axis tick labels
    ax1.tick_params(axis='y', pad=10)  # Increase padding for y-axis tick labels
    ax1.tick_params(axis='z', pad=8)  # Increase padding for z-axis tick labels
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)', fontsize=22, labelpad=27)
    ax1.set_ylabel('Y (mm)', fontsize=22, labelpad=30)
    ax1.set_zlabel('Z (mm)', fontsize=22, labelpad=17)
    ax1.set_title(f'3-D reconstruction', fontsize=28, pad=1)
    ax1.view_init(elev=-15., azim=210., roll=180.)    
    im_side = plt.imread(im_files_side[frame_id])[:,:,:3][::-1, ::-1]
    im_side = im_side / im_side.max() * 255
    im_side = im_side.astype(np.uint8)
    im_top = plt.imread(im_files_top[frame_id])[:,:,:3]
    im_top = im_top / im_top.max() * 255
    im_top = im_top.astype(np.uint8)
    im_side = draw_axes(im_side, center_x=150, center_y=150, length=100, color=(255,255,255), axis1_name='Y', axis2_name='Z')
    im_top = draw_axes(im_top, center_x=150, center_y=150, length=100, color=(255,255,255), axis1_name='X', axis2_name='-Y')

    im = np.concatenate((im_top, 255 * np.ones((50, im_side.shape[1], im_side.shape[2]), np.uint8), im_side), axis=0)
    font = cv.FONT_HERSHEY_SIMPLEX    
    font_thickness = 3
    ax2.cla()
    cv.putText(im, f'Time: {1 / frame_rate * (frame_id-1):.2f} ms', (650, 100), font, 3, (255,255,255), font_thickness, cv.LINE_AA)
    ax2.imshow(im, cmap='gray')
    ax2.scatter(C_top[0, frame_id], 
                (C_top[1, frame_id]), 
                s=20, color='tab:red') 
    ax2.scatter(im_side.shape[1] - C_side[0, frame_id], 
                im_side.shape[0] - (C_side[1, frame_id]) + pad + im_top.shape[0], 
                s=20, color='tab:red')
    ax2.set_xticks([])
    ax2.set_yticks([])
    

#%% 
principal_point_pixel_cam_0 = torch.zeros(2).to(torch.float64)
principal_point_pixel_cam_1 = torch.zeros_like(principal_point_pixel_cam_0)
focal_length_cam_1 = torch.tensor(0.).to(torch.float64)
focal_length_cam_2 = torch.zeros_like(focal_length_cam_1)
R = torch.eye(3).to(torch.float64)
T = torch.zeros(3,1).to(torch.float64)
tank_angles = torch.zeros(3).to(torch.float64)
tank_center = torch.zeros_like(tank_angles)
tank_thickness = torch.tensor([0.]).to(torch.float64)
tank_size = torch.zeros(3).to(torch.float64)
refractive_index_acrylic = torch.tensor([0]).to(torch.float64)
refractive_index_water = torch.zeros_like(refractive_index_acrylic)

arena = Arena_fish_tank_pairwise_distances(principal_point_pixel_cam_0,
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

model_checkpoint_dir = f'/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2025_1_21_22_36_30_fish'
PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena.load_state_dict(checkpoint['model_state_dict'])

#%%
pixels_two_cams = torch.vstack((C_top, C_top, C_side, C_side))
recon_3D, closest_distance_loss_test, pairwise_distance_recon_test, _, _ = arena(pixels_two_cams)
recon_3D = recon_3D[:, :pixels_two_cams.shape[1] // 2].detach().numpy()
x_min, x_max, y_min, y_max, z_min, z_max = get_plot_limits(recon_3D, zoom_factor=1)
num_annotations = recon_3D.shape[1]

if save_video:
    ani = animation.FuncAnimation(
        fig, update, frames=num_annotations-1, 
        interval=100, repeat=False,
        )
    # Save the animation as a video file
    ani.save(f'{output_video_path}.mp4', writer='ffmpeg', fps=6)