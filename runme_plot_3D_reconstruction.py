import openpyxl
import torch
from arenasEfficient import Arena_single_camera_prism_grid_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2 
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
import numpy as np

def draw_axes(image, center_x, center_y, length, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = length / 75
    font_thickness = 2

    # Draw the x-axis
    cv2.line(image, (center_x, center_y), (center_x + length, center_y), color, 2)

    # Draw the y-axis
    cv2.line(image, (center_x, center_y), (center_x, center_y + length), color, 2)

    # Add arrowheads to the axes
    arrow_length = length // 15
    cv2.arrowedLine(image, (center_x + length, center_y), (center_x + length - arrow_length, center_y - arrow_length), color, 2)
    cv2.arrowedLine(image, (center_x, center_y + length), (center_x + arrow_length, center_y + length - arrow_length), color, 2)
    cv2.putText(image, 'Y', (center_x + length + length // 6, center_y - length // 6), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Label for Y-axis
    cv2.putText(image, 'X', (center_x - length // 2, center_y + length + length // 4), font, font_scale, color, font_thickness, cv2.LINE_AA)
    return image

excel_file_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_17/fly_images/cam_0/image_cam_0_date_2024_12_20_time_16_40_17_v001_annotations.xlsx'

#%%
# Load data
model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2024_12_27_12_34_37'
#2024_12_26_12_53_16
output_video_path = excel_file_path.split('/')[-1].split('.')[0]
save_video = True

PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
checkpoint = torch.load(PATH, weights_only=True)
arena = Arena_single_camera_prism_grid_distance(
            principal_point_pixel_cam_0=torch.tensor([0.,0.]).to(torch.float64),
            principal_point_pixel_cam_1=torch.tensor([0.,0.]).to(torch.float64), 
            focal_length_cam_0=torch.tensor(0.).to(torch.float64),
            focal_length_cam_1=torch.tensor(0.).to(torch.float64),
            R_stereo_cam=None,
            T_stereo_cam=None, 
            prism_angles=torch.tensor([0.,0.,0.]).to(torch.float64),
            prism_center=torch.tensor([0.,0.,0.]).to(torch.float64).unsqueeze(-1),
            prism_size=20.,
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
for cell in sheet.iter_rows(min_row=1, max_row=num_rows, values_only=True):
    rows.append(cell)  # This will print a tuple of values in the specified row

# Add code to register keypoint ids here
rows = rows[1:]
for row in rows:
    if None in row:
        rows.remove(row)

num_keypoints = len(rows[0]) - 1
num_annotations = len(rows)

real_pixels_cam_0 = torch.zeros((2, num_annotations, num_keypoints)).to(torch.float64)
virtual_pixels_cam_0 = torch.zeros_like(real_pixels_cam_0)

im_files = []
for i, row in enumerate(rows):
    if (i == 0):
        continue
    im_files.append(row[0])
    row = row[1:]
    for kpt_id in range(num_keypoints):
        ann = torch.tensor([float(el) for el in row[kpt_id].split(",")]).to(torch.float64)
        real_pixels_cam_0[:, i, kpt_id] = ann[:2]
        virtual_pixels_cam_0[:, i, kpt_id] = ann[2:]

with torch.no_grad():
    real_pixels_cam_0_ = real_pixels_cam_0.flatten(1)
    virtual_pixels_cam_0_ = virtual_pixels_cam_0.flatten(1)
    recon_3D, closest_distance, recon_distorted_pixels_1, recon_real_loss, pairwise_distance_recon, intersection_penalty, distortion_penalty = arena(real_pixels_cam_0_.repeat(2,1), virtual_pixels_cam_0_.repeat(2,1)) # repeating to account for the pairwise distance computation in the arena model
    recon_3D = recon_3D[:, :recon_3D.shape[1]//2] # discarding repeating second half
    recon_3D = recon_3D.view(3, num_annotations, num_keypoints)

fig = plt.figure(figsize=(30,20))
#ax = fig.add_subplot(projection='3d')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # Equal width for both columns
# Create subplots
ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Larger subplot spanning both columns
ax2 = fig.add_subplot(gs[0, 1])  # Smaller subplot on the left

links = [[6, 0], [6, 1], [7, 2], [7, 3], [8, 4], [8, 5], [6, 7, 8], [9, 6]]
links_colors = ['r', 'r', 'b', 'b', 'm', 'm', 'gray', 'k']
link_linewidths = [1, 1, 1, 1, 1, 1, 1, 2]
recon_3D = recon_3D[:, 1:, :].numpy()

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

def update(frame_id):
    ax1.cla()
    ax1.set_title(f'3-D reconstruction', fontsize=24)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax2.cla()
    ax1.scatter(recon_3D[0, frame_id, :],
               recon_3D[1, frame_id, :],
               recon_3D[2, frame_id, :],
               s=2,
               color='r')
    ax1.scatter(recon_3D[0, frame_id, 9],
               recon_3D[1, frame_id, 9],
               recon_3D[2, frame_id, 9],
               s=1,
               marker='o',
               color='r')
    for link in links:
        ax1.plot(recon_3D[0, frame_id, link],
                recon_3D[1, frame_id, link],
                recon_3D[2, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=link_linewidths[links.index(link)]
        )
    
    ax1.set_aspect('equal', adjustable='box')
    ax1.view_init(elev=-70., azim=1., roll=0.)
    plt.draw()
    
    im = plt.imread(im_files[frame_id-1]).T
    im = cv2.flip(im, 0)
    im = np.stack((im,) * 3, axis=-1)
    im_height, im_width, _ = im.shape
    axis_origin = [im_width * 4 // 5, im_height // 20]
    im = draw_axes(
        im, 
        axis_origin[0],
        axis_origin[1],
        100,
        (255, 255, 255),
    )
    ax2.imshow(im)

    ax2.scatter(virtual_pixels_cam_0[1, frame_id, :], 
                im_height - virtual_pixels_cam_0[0, frame_id, :], 
                s=1, color='g')
    ax2.scatter(real_pixels_cam_0[1, frame_id, :], 
                im_height - real_pixels_cam_0[0, frame_id, :], 
                s=1, color='r')
    
    for link in links:
        ax2.plot(virtual_pixels_cam_0[1, frame_id, link],
                im_height - virtual_pixels_cam_0[0, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=link_linewidths[links.index(link)] / 2
        )
        ax2.plot(real_pixels_cam_0[1, frame_id, link],
                im_height - real_pixels_cam_0[0, frame_id, link],
                color=links_colors[links.index(link)],
                linewidth=link_linewidths[links.index(link)] / 2
        )

    ax2.grid(False)
    ax2.set_title('Manual labels over raw data', fontsize=22)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.suptitle(f'Frame_id: {frame_id}', fontsize=32)
    plt.draw()

#num_annotations = 101
if not save_video:
    for frame_id in range(1, num_annotations - 1):  
        update(frame_id)
        plt.pause(0.01)

if save_video:
    ani = animation.FuncAnimation(
        fig, update, frames=num_annotations-1, 
        interval=100, repeat=False,
        )
    # Save the animation as a video file
    ani.save(f'{output_video_path}.mp4', writer='ffmpeg', fps=10)

# Close the workbook (optional, as it will be closed automatically when the program ends)
workbook.close()


# %%
