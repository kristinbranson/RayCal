import matplotlib.pyplot as plt
import numpy as np
import torch
from arenasEfficient import Arena_single_camera_prism_grid_distance
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import openpyxl

#%%
# Load  data
model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2024_12_18_18_1_22'
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
images_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_13/fly_images/cam_0/image_cam_0_date_2024_12_17_time_21_02_45_v001_selected'

# %% Click event
coordinates = []
global flag # To continue with the same keypoint. flag = 0 to move to the next keypoint
flag = 1
def onclick(event):
    global flag
    # Check if the click is within the image boundaries
    if event.button == 3:
        print('Right click. Exiting the annotation')
        flag = 0
        
    if event.xdata is not None and event.ydata is not None and event.button == 1:
        # Store the (x, y) coordinates
        coordinates.append((event.xdata, event.ydata))
        print(f"Clicked at: ({event.xdata}, {event.ydata})")

# %% 
def convert_annotations_to_excel(real_annotation, virtual_annotation, frame_id, kpt_id):
    if (len(real_annotation.shape) > 1):
        real_annotation = real_annotation.squeeze()

    if (len(virtual_annotation.shape) > 1):
        virtual_annotation = virtual_annotation.squeeze()

    data = torch.hstack((real_annotation, virtual_annotation)).detach().numpy().tolist()
    annotations = {}
    annotations['frame_id'] = frame_id
    annotations['keypoint_id'] = kpt_id
    annotations['appended_coordinates'] = data
    return annotations

# %%
def append_to_excel(file_name, annotations, keypoints_dict):
    num_keypoints = len(keypoints_dict)
    data = annotations['appended_coordinates']
    frame_id = annotations['frame_id']
    kpt_id = annotations['keypoint_id']

    try:
        # Load the existing workbook
        workbook = openpyxl.load_workbook(file_name)
        sheet = workbook.active  # Get the active sheet

    except FileNotFoundError:
        # If the file does not exist, create a new workbook and sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active

    # Find the next empty row
    next_row = sheet.max_row 
    print(next_row)
    if next_row == 1:
        # If the sheet is empty, add the column headers
        sheet.cell(row=1, column=1, value="Frame")
        column_headers = ["Frame no."]
        column_headers = column_headers + [keypoint_name for keypoint_name in keypoints_dict]
        print('Adding header to the excel file: {column_headers}')
        for col_num, header in enumerate(column_headers, start=1):
            sheet.cell(row=1, column=col_num, value=header)
        next_row = 2

    # Append data to the next empty row    
    value_str = ', '.join(map(str, data))
    sheet.cell(row=next_row, column=1, value=frame_id)
    sheet.cell(row=next_row, column=kpt_id+2, value=value_str) # Plus 2, because the first column is the frame number, and kpt has 0 indexing

    # Save the workbook
    workbook.save(file_name)
    print(f"Data appended to {file_name} successfully.")


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
def get_user_annotations(arena, image_folder, keypoints_dict, first_frame=0, excel_file_path=None):
    num_keypoints = len(keypoints_dict)
    # List all image files in the specified folder
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', 'bmp'))]
    print(f'Found {len(image_files)} images in the folder: {image_folder}')
    print(f'Starting annotation from image: {image_files[first_frame]}, which is the {first_frame}th image')
    image_files = image_files[first_frame:]
    virtual_annotations = torch.zeros(len(image_files), 2, num_keypoints).to(torch.float64)
    real_annotations = torch.zeros_like(virtual_annotations)

    
    for image_id, image_file in enumerate(image_files):
        # Load the image
        global flag
        image_path = os.path.join(image_folder, image_file)
        img = plt.imread(image_path)
        
        for kpt_id in range(num_keypoints):
            # Show the image
            # Create a new figure and axis            
            fig, ax = plt.subplots(figsize=(22, 20))
            ax.imshow(img, cmap='gray')            
            
            # Get virtual annotations
            # Connect the click event to the onclick function
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            scat1 = None
            scat2 = None
            ax.set_title(f"Click on the virtual image: {image_file} to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            plt.show()
            len_coor = 0
            while True:                
                if flag == 0:
                    flag = 1
                    break    
                plt.pause(0.05)
            
                if len(coordinates) == len_coor:
                    continue
                # After closing the image, save the coordinates
                virtual_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                coordinates.clear()

                # Get real annotations
                annotations_curve = get_annotations_curve(arena, virtual_annotations_frame)
                ax.imshow(img, cmap='gray')
                

                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                scat1 = ax.scatter(
                    virtual_annotations_frame[0,:].detach().numpy(),
                    virtual_annotations_frame[1,:].detach().numpy(),
                    s=2,
                    color='g',
                )
                scat2 = ax.scatter(
                    annotations_curve[0,:].detach().numpy(),
                    annotations_curve[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.2
                )
                len_coor = len(coordinates)
                #ax.legend(['User annotations in virtual view', 'Predicted annotations in real view'])
                plt.draw()
                    
            
            # Get plausible real annotations again just to be sure
            annotations_curve = get_annotations_curve(arena, virtual_annotations_frame)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Click on the real image: {image_file} to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            
            while True:
                if flag == 0:
                    flag = 1
                    break
                plt.pause(0.1)
                
                if len_coor == len(coordinates):
                    continue

                scat1 = ax.scatter(
                        annotations_curve[0,:].detach().numpy(),
                        annotations_curve[1,:].detach().numpy(),
                        s=0.1,
                        color='r',
                        alpha=0.2
                    )
                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                real_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                coordinates.clear()
                annotations_curve = get_annotations_curve(arena, real_annotations_frame)
                ax.imshow(img, cmap='gray')
                
                scat2 =  ax.scatter(
                    real_annotations_frame[0,:].detach().numpy(),
                    real_annotations_frame[1,:].detach().numpy(),
                    s=2,
                    color='r',
                )                
                len_coor = len(coordinates)
                plt.draw()

            
            # Clear the coordinates for the next image
            coordinates.clear()
            annotations = convert_annotations_to_excel(real_annotations_frame, virtual_annotations_frame, frame_id=first_frame + image_id, kpt_id=kpt_id)
            if excel_file_path is not None:
                append_to_excel(excel_file_path, annotations, keypoints_dict)
            else: 
                print("Excel file path not provided. Annotations not saved.")
            virtual_annotations[image_id, :, kpt_id] = virtual_annotations_frame[:,0]
            real_annotations[image_id, :, kpt_id] = real_annotations_frame[:,0]

            plt.close('all')
            flag = 1
    
    return virtual_annotations, real_annotations

# %%
excel_file_name='annotations.xlsx'
keypoints_dict = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'Head', 'Belly', 'Thorax Tip']
excel_file_path = os.path.join(images_dir, excel_file_name)
virtual_annotations, real_annotations = get_user_annotations(arena, images_dir, keypoints_dict, first_frame=0, excel_file_path=excel_file_path)
# %%
