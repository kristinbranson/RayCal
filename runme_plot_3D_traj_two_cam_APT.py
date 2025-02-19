import openpyxl
import torch
from arenasEfficient import Arena_reprojection_loss_two_cameras_prism_grid_distances
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2 
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
import numpy as np
import os

def draw_axes(image, center_x, center_y, length, color, axis1_name, axis2_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = length / 75
    font_thickness = 2

    # Draw the y-axis
    cv2.line(image, (center_x, center_y), (center_x + length, center_y), color, 2)

    # Draw the x-axis
    cv2.line(image, (center_x, center_y), (center_x, center_y - length), color, 2)

    # Add arrowheads to the axes
    arrow_length = length // 5
    cv2.arrowedLine(image, (center_x + length - arrow_length, center_y), (center_x + length, center_y), color, 2, tipLength=0.5)
    cv2.arrowedLine(image, (center_x, center_y - length + arrow_length), (center_x, center_y - length), color, 2, tipLength=0.5)
    cv2.putText(image, axis1_name, (center_x + length + length // 6, center_y + length // 6), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Label for Y-axis
    cv2.putText(image, axis2_name, (center_x - length // 2, center_y - length + length // 4), font, font_scale, color, font_thickness, cv2.LINE_AA)
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

excel_file_path = '/groups/branson/bransonlab/aniket/APT/exp20v2_labels.xlsx'

#%%
# Load data
model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/exp_20_2025_2_11_17_51_27'
#2024_12_26_12_53_16
output_video_path = excel_file_path.split('.')[0]
save_video = True
num_views = 2

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena = Arena_reprojection_loss_two_cameras_prism_grid_distances(
            principal_point_pixel_cam_0=torch.tensor([0.,0.]).to(torch.float64),
            principal_point_pixel_cam_1=torch.tensor([0.,0.]).to(torch.float64), 
            focal_length_cam_0=torch.tensor(0.).to(torch.float64), 
            focal_length_cam_1=torch.tensor(0.).to(torch.float64),
            R_stereo_cam=torch.eye(3).to(torch.float64),
            T_stereo_cam=torch.zeros(3,1).to(torch.float64), 
            prism_angles=torch.zeros(3,).to(torch.float64), 
            prism_center=torch.zeros(3,1).to(torch.float64), 
            )

arena.load_state_dict(checkpoint['model_state_dict'])

# Load the workbook
workbook = openpyxl.load_workbook(excel_file_path)

# Select a worksheet
sheet = workbook.active  # or workbook['SheetName'] to select a specific sheet
num_rows = sheet.max_row



# Read data from a range of cells
row_number = 1
rows = []
for cell in sheet.iter_rows(min_row=0, max_row=num_rows, values_only=True):
    rows.append(cell)  # This will print a tuple of values in the specified row


# Find number of keypoints
num_keypoints = 0
flag = 0 # for finding the first index with keypoint
for i, el in enumerate(rows[0]):
    if 'p_' in el:
        num_keypoints += 1
        if flag == 0:
            first_kpt_id = i
            flag = 1

last_kpt_id = i + num_keypoints
num_keypoints = num_keypoints // num_views // 2 #(divide by 2 to account for (x,y) pairs)


# Add code to register keypoint ids here
rows = rows[1:]
num_annotations = len(rows)

real_pixels_cam_0 = torch.zeros((2, num_annotations, num_keypoints)).to(torch.float64)
virtual_pixels_cam_0 = torch.zeros_like(real_pixels_cam_0)
real_pixels_cam_1 = torch.zeros_like(real_pixels_cam_0)
virtual_pixels_cam_1 = torch.zeros_like(real_pixels_cam_1)

frames = []
im_files = []
for i, row in enumerate(rows):
    im_files.append(row[0])
    frames.append(row[2])
    for kpt_id in range(2 * num_keypoints):
        # Subtract 1 to account for MATLAB indexing
        ann = - 1 +  torch.tensor([float(el) for el in [row[first_kpt_id + kpt_id], row[first_kpt_id + num_views * num_keypoints + kpt_id]]]) # APT stores x first, then y
        # [kpt_id].split(",")]).to(torch.float64)
        if kpt_id < num_keypoints:
            real_pixels_cam_0[:, i, kpt_id] = ann 
            virtual_pixels_cam_0[:, i, kpt_id] = ann 
        else:
            real_pixels_cam_1[:, i, kpt_id - num_keypoints] = ann
            virtual_pixels_cam_1[:, i, kpt_id - num_keypoints] = ann

with torch.no_grad():
    real_pixels_cam_0_ = real_pixels_cam_0.flatten(1).repeat(2,1)
    virtual_pixels_cam_0_ = virtual_pixels_cam_0.flatten(1).repeat(2,1)
    real_pixels_cam_1_ = real_pixels_cam_1.flatten(1).repeat(2,1)
    virtual_pixels_cam_1_ = virtual_pixels_cam_1.flatten(1).repeat(2,1)
    virtual_pixels_two_cams = torch.vstack((virtual_pixels_cam_1_, virtual_pixels_cam_0_))
    real_pixels_two_cams = torch.vstack((real_pixels_cam_0_, virtual_pixels_cam_1_))
    recon_3D, closest_distance, closest_distance_virtual, recon_pixels_1, recon_pixels_2, recon_3D_real, recon_3D_virtual, distortion_penalty_cam_0, distortion_penalty_cam_1, intersection_penalty_1, intersection_penalty_2, pairwise_distance = arena(
        virtual_pixels_two_cams, 
        real_pixels_two_cams,
    )
    recon_3D = recon_3D_virtual[:, :recon_3D_virtual.shape[1] // 2] # discarding repeating second half
#    recon_3D = recon_3D[:, :recon_3D.shape[1]//2] # discarding repeating second half 
    recon_3D = recon_3D.view(3, num_annotations, num_keypoints)

cam_0_im_path = '/groups/branson/bransonlab/aniket/APT/deepnet/images0/'
cam_1_im_path = '/groups/branson/bransonlab/aniket/APT/deepnet/images1/'
im_files_cam_0 = [os.path.join(cam_0_im_path, filename) for filename in os.listdir(cam_0_im_path)]
im_files_cam_1 = [os.path.join(cam_1_im_path, filename) for filename in os.listdir(cam_1_im_path)]


fig = plt.figure(figsize=(40, 18))
#ax = fig.add_subplot(projection='3d')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])  # Equal width for both columns
# Create subplots
ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Larger subplot spanning both columns
ax2 = fig.add_subplot(gs[0, 1])  # Smaller subplot on the left

links = [[i, i + 6, i + 12] for i in range(6)]
links_colors = ['tab:blue', 'tab:orange', 'tab:green', 
                'tab:blue', 'tab:orange', 'tab:green',
                ]
link_linewidths = [1, 1, 1, 1, 1, 1, 1, 1]
link_linewidths = [3 * width for width in link_linewidths]

recon_3D = recon_3D[:, :, :].numpy()
x_min, x_max, y_min, y_max, z_min, z_max = get_plot_limits(recon_3D, zoom_factor=0.5)



def process_img_for_plot(im, split_and_stack=False):
    """
    Crop image along a horizontal line defined using 'dividing_line' and append the top half to the bottom half
    Returns the processed image and the offsets in y-coordinate that would need to be added to any coordinates plotted over the processed image for the top half and bottom half
    """
    im = cv2.flip(im, 0)
    im = np.stack((im,) * 3, axis=-1)
    im_height, im_width, _ = im.shape    
    axis_origin = [im_width * 4 // 5, im_height // 10]
    im = draw_axes(
        im, 
        axis_origin[0],
        axis_origin[1],
        100,
        (255, 255, 255),
        'Y',
        'X',
    )
    im = draw_axes(
        im,
        axis_origin[0],
        axis_origin[1] + im_height // 2,
        100,
        (255, 255, 255),
        'Y',
        'Z',
    )

    if not split_and_stack:
        return im, 0, 0
    else:
        dividing_line = im_height * 5 // 10 # For cropping and appending images below
        im2 = im[dividing_line:, ...]
        im1 = np.zeros_like(im2)
        im1[im2.shape[0] - dividing_line:, ...] = im[:dividing_line, ...]
        im = np.concatenate((im1, im2), axis=1)
        im1_Y_offset = -(im2.shape[0])
        im2_X_offset = im1.shape[1]
    return im, im1_Y_offset, im2_X_offset


def update(frame_id):
    ax1.cla()
    # Get the current limits
    # Set the separation for ticks
    x_tick_step = 2
    y_tick_step = 2
    z_tick_step = 2

    # Create new ticks based on the limits and step
    x_ticks = np.arange(np.floor(x_min), np.ceil(x_max) + x_tick_step, x_tick_step)
    y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + y_tick_step, y_tick_step)
    z_ticks = np.arange(np.floor(z_min), np.ceil(z_max) + z_tick_step, z_tick_step)

    print(x_min, x_max, y_min, y_max, z_min, z_max)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, fontsize=18)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks, fontsize=18)
    ax1.set_zticks(z_ticks)
    ax1.set_zticklabels(z_ticks, fontsize=18)
    ax1.tick_params(axis='x', pad=15)  # Increase padding for x-axis tick labels
    ax1.tick_params(axis='y', pad=15)  # Increase padding for y-axis tick labels
    ax1.tick_params(axis='z', pad=15)  # Increase padding for z-axis tick labels
    ax1.tick_params(axis='both', width=0)
    ax1.set_xlabel('X (mm)', fontsize=18, labelpad=26)  # Increase padding for x-axis label
    ax1.set_ylabel('Y (mm)', fontsize=18, labelpad=26)  # Increase padding for y-axis label
    ax1.set_zlabel('Z (mm)', fontsize=18, labelpad=26)  # Increase padding for z-axis label
    ax1.set_title(f'3-D reconstruction', fontsize=28, pad=1)

    
    ax1.scatter(recon_3D[0, frame_id, :],
               recon_3D[1, frame_id, :],
               recon_3D[2, frame_id, :],
               s=15,
               color='tab:gray')

    for link in links:
        ax1.plot(recon_3D[0, frame_id, link],
                recon_3D[1, frame_id, link],
                recon_3D[2, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=link_linewidths[links.index(link)]
        )
    
    ax1.set_aspect('equal', adjustable='box')
    ax1.view_init(elev=-30., azim=-70., roll=80.)
    
    #im = plt.imread(im_files[frame_id-1]).T
    print(f'Reading frame {im_files_cam_0[frames[frame_id] - 1]}')
    im1 = plt.imread(im_files_cam_0[frames[frame_id] - 1])
    im1 = np.stack((im1,) * 3, axis=-1)
    im2 = plt.imread(im_files_cam_1[frames[frame_id] - 1])
    im2 = np.stack((im2,) * 3, axis=-1)
    im = np.concatenate((im1, im2), axis=1)
    im2_offset_X = im1.shape[1]
    #im = np.ones((1920, 1280), dtype=np.float16)
    
    #im, real_offset_Y, virtual_offset_X = process_img_for_plot(im, split_and_stack=True)
    virtual_offset_Y = 0
    im_height, im_width, _ = im.shape

    font = cv2.FONT_HERSHEY_SIMPLEX    
    font_thickness = 3  
    #im = np.concatenate((im, np.ones((10, im.shape[1], #im.shape[2])), im), axis=0)
    ax2.cla()
    cv2.putText(im, f'Time: {13 * (frame_id-1)} ms', (75, 100), font, 3, (255,255,255), font_thickness, cv2.LINE_AA)
    ax2.imshow(im)
    

    ax2.scatter(virtual_pixels_cam_0[0, frame_id, :] + im2_offset_X, 
                virtual_pixels_cam_0[1, frame_id, :], 
                s=10, color='tab:gray')
    ax2.scatter(virtual_pixels_cam_1[0, frame_id, :],
                virtual_pixels_cam_1[1, frame_id, :], 
                s=10, color='tab:gray')
    
                
    for link in links:
        ax2.plot(virtual_pixels_cam_0[0, frame_id, link] + im2_offset_X,
                virtual_pixels_cam_0[1, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=link_linewidths[links.index(link)] 
        )
        ax2.plot(virtual_pixels_cam_1[0, frame_id, link] ,
                virtual_pixels_cam_1[1, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=0.75*link_linewidths[links.index(link)]
        )
    
    ax2.grid(False)
    ax2.set_title('Manual labels over raw data', fontsize=28, pad=120)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.subplots_adjust(top = 0.85, bottom = 0, right = 0.96, left = 0, hspace = 0, wspace = 0)  
    plt.draw()


if not save_video:
    for frame_id in range(0, num_annotations):  
        update(frame_id)
        plt.pause(0.01)

#num_annotations = 1
if save_video:
    ani = animation.FuncAnimation(
        fig, update, frames=num_annotations, 
        interval=100, repeat=False,
        )
    # Save the animation as a video file
    ani.save(f'{output_video_path}.mp4', writer='ffmpeg', fps=6)

# Close the workbook (optional, as it will be closed automatically when the program ends)
workbook.close()

# %%
