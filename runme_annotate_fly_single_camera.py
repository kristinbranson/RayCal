import matplotlib.pyplot as plt
import numpy as np
import torch
from arenasEfficient import Arena_single_camera_prism_grid_distance
import scipy.io as sio
import os
from matplotlib.widgets import Cursor

#%%
# Load  data
model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2024_12_17_15_0_2'
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
images_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_9/fly_images/cam_0/'

# %% Click event
coordinates = []
def onclick(event):
    # Check if the click is within the image boundaries
    if event.xdata is not None and event.ydata is not None:
        # Store the (x, y) coordinates
        coordinates.append((event.xdata, event.ydata))
        print(f"Clicked at: ({event.xdata}, {event.ydata})")


# %%
def get_annotations_curve(arena, user_annotation):
    undistorted_annotations = arena.camera1.undistort_pixels_classical(user_annotation[:2],
    arena.radial_dist_coeffs)
    cam_1_ray = arena.camera1(undistorted_annotations)
    _, _, emergent_ray, _ = arena.prism(cam_1_ray)
    origin = emergent_ray.origin[:,0][:,None]
    direction = emergent_ray.direction[:,0][:,None]
    s = torch.linspace(0, 3, 100).to(torch.float64)
    r = origin + s[None, :] * direction

    R1 = torch.eye(3, 3).to(torch.float64)
    T1 = torch.zeros(3, 1).to(torch.float64)
    annotations_curve = arena.camera1.reproject(torch.tensor(r).to(torch.float64), R1, T1)
    return annotations_curve

# %%
def get_user_annotations(arena, image_folder, num_keypoints, first_frame=0):
    # List all image files in the specified folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', 'bmp'))]
    print(f'Found {len(image_files)} images in the folder: {image_folder}')
    print(f'Starting annotation from image: {image_files[first_frame]}, which is the {first_frame}th image')
    virtual_annotations = torch.zeros(len(image_files), 2, num_keypoints).to(torch.float64)
    real_annotations = torch.zeros_like(virtual_annotations)

    
    for image_id, image_file in enumerate(image_files):
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        img = plt.imread(image_path)

        # Create a new figure and axis
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Click on the image: {image_file} to annotate {num_keypoints} keypoints")
        
        # Get virtual annotations
        # Connect the click event to the onclick function
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Show the image
        plt.show(block=True)

        # After closing the image, save the coordinates
        virtual_annotations_frame = torch.tensor(coordinates).to(torch.float64).T
        coordinates.clear()
        print(virtual_annotations_frame.shape)

        # Get real annotations
        annotations_curve = get_annotations_curve(arena, virtual_annotations_frame)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Click on the image: {image_file} to annotate {num_keypoints} keypoints")
        plt.scatter(
            virtual_annotations_frame[0,:].detach().numpy(),
            virtual_annotations_frame[1,:].detach().numpy(),
            s=2,
            color='g',
        )
        plt.scatter(
            annotations_curve[0,:].detach().numpy(),
            annotations_curve[1,:].detach().numpy(),
            s=0.1,
            color='r',
            alpha=0.2
        )
        plt.legend(['User annotations in virtual view', 'Predicted annotations in real view'])
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show(block=True)

        # After closing the image, save the coordinates
        real_annotations_frame = torch.tensor(coordinates).to(torch.float64).T
        
        # Clear the coordinates for the next image
        coordinates.clear()
        virtual_annotations[image_id, ...] = virtual_annotations_frame
        real_annotations[image_id, ...] = real_annotations_frame
    
    return virtual_annotations, real_annotations

# %%
virtual_annotations, real_annotations = get_user_annotations(arena, images_dir, num_keypoints=1, first_frame=0)