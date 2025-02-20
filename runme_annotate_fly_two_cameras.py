#%%
# Load  data
import matplotlib.pyplot as plt
import numpy as np
import torch
from arenasEfficient import Arena_reprojection_loss_two_cameras_prism_grid_distances
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import openpyxl
from ray_tracing_simulator_nnModules_grad import get_rot_mat
from skimage import exposure

#%%
# Load  data
model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/exp_20_2025_2_11_17_51_27'
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
experiment_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_20/fly_images/'
images_dir = experiment_dir # NOTE: This is being done slightly differently compared with the single camera case

# %% Click event
coordinates = []
global flag # To continue with the same keypoint. flag = 0 to move to the next keypoint
flag = 1
def onclick(event):
    global flag
    # Check if the click is within the image boundaries
    if event.button == 3:
        #print('Right click. Exiting the annotation')
        flag = 0
        
    if event.xdata is not None and event.ydata is not None and event.button == 1:
        # Store the (x, y) coordinates
        coordinates.append((event.xdata, event.ydata))
        #print(f"Clicked at: ({event.xdata}, {event.ydata})")

# %% 
def convert_annotations_to_excel(real_annotation, virtual_annotation, im_id, kpt_id):
    if (len(real_annotation.shape) > 1):
        real_annotation = real_annotation.squeeze()

    if (len(virtual_annotation.shape) > 1):
        virtual_annotation = virtual_annotation.squeeze()

    data = torch.hstack((real_annotation, virtual_annotation)).detach().numpy().tolist()
    annotations = {}
    annotations['frame_id'] = im_id
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
    if kpt_id == 0:
        next_row = sheet.max_row + 1
    else:
        next_row = sheet.max_row
    if next_row == 2:
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
def get_epipolar_line(arena, user_annotation, 
                          cam_label):
    if "primary" in cam_label:
        labelled_camera = arena.camera1
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_0
        unlabelled_camera = get_secondary_camera(arena)
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        R_labelled = torch.eye(3, 3).to(torch.float64)
        T_labelled = torch.zeros(3, 1).to(torch.float64)
        R_unlabelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_unlabelled = arena.T_stereo_cam

    elif "secondary" in cam_label:
        labelled_camera = get_secondary_camera(arena)
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        unlabelled_camera = arena.camera1
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_0    
        R_unlabelled = torch.eye(3, 3).to(torch.float64)
        T_unlabelled = torch.zeros(3, 1).to(torch.float64)
        R_labelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_labelled = arena.T_stereo_cam

    else:
        print("Please provide appropriate camera labels")


    if cam_label in ["primary_virtual", "secondary_virtual"]:
        with torch.no_grad():
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            
            cam_1_ray = labelled_camera(undistorted_annotations)
            _, _, emergent_ray, _ = arena.prism(cam_1_ray)
            origin = emergent_ray.origin[:,0][:,None]
            direction = emergent_ray.direction[:,0][:,None]
            s = torch.linspace(0, 8, 50).to(torch.float64)
            r = origin + s[None, :] * direction       

            
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(torch.tensor(r).to(torch.float64), R_unlabelled, T_unlabelled)
            annotations_curve_labelled_camera = labelled_camera.reproject(torch.tensor(r).to(torch.float64), R_labelled, T_labelled)
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            annotations_curve_labelled_camera = labelled_camera.distort_pixels_classical(
                                                                annotations_curve_labelled_camera,
                                                                cam_labelled_dist_coeff)
            return annotations_curve_unlabelled_camera, annotations_curve_labelled_camera
    

    elif cam_label in ["primary_real", "secondary_real"]:
        with torch.no_grad():
            prism_distance = torch.norm(arena.prism_center)
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            cam_1_ray = labelled_camera(undistorted_annotations)
            origin = cam_1_ray.origin[:,0][:,None]
            direction = cam_1_ray.direction[:,0][:,None]
            s = torch.linspace(prism_distance, prism_distance + 10, 50).to(torch.float64)
            r = origin + s[None, :] * direction        
            
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(torch.tensor(r).to(torch.float64), R_unlabelled, T_unlabelled)            
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            return annotations_curve_unlabelled_camera

    
# %%
def get_user_annotations(arena, image_folder, keypoints_dict, first_frame=0, excel_file_path=None, step=1):
    num_keypoints = len(keypoints_dict)
    # List all image files in the specified folder
    image_folder_primary = os.path.join(image_folder, 'cam_0')
    image_folder_primary = [os.path.join(image_folder_primary, filename) for filename in os.listdir(image_folder_primary) if filename.startswith("image") and not filename.endswith('ufmf')][0]

    image_folder_secondary = os.path.join(image_folder, 'cam_1')
    image_folder_secondary = [os.path.join(image_folder_secondary, filename) for filename in os.listdir(image_folder_secondary) if filename.startswith("image") and not filename.endswith('ufmf')][0]
    image_files_primary = [f for f in os.listdir(image_folder_primary) if f.endswith(('.png', 'bmp'))]
    image_files_secondary = [f for f in os.listdir(image_folder_secondary) if f.endswith(('.png', 'bmp'))]

    print(f'Found {len(image_files_primary)} images in the folder: {image_folder}')
    print(f'Starting annotation from image: {image_files_primary[first_frame]}, which is the {first_frame}th image')
    image_files_primary = image_files_primary[first_frame::step]
    image_files_secondary = image_files_secondary[first_frame::step]
    primary_virtual_annotations = torch.zeros(len(image_files_primary), 2, num_keypoints).to(torch.float64)
    secondary_virtual_annotations = torch.zeros_like(primary_virtual_annotations)
    primary_real_annotations = torch.zeros_like(primary_virtual_annotations)
    secondary_real_annotations = torch.zeros_like(primary_virtual_annotations)

    for image_id, (image_file_primary, image_file) in enumerate(zip(image_files_primary, image_files_secondary)):
        # Load the image
        global flag
        image_path_primary = os.path.join(image_folder_primary, image_file)
        image_path_secondary = os.path.join(image_folder_secondary, image_file)

        img_primary = plt.imread(image_path_primary)
        img_secondary = plt.imread(image_path_secondary)
        img = np.hstack((img_primary,
                        img_secondary))
        img = exposure.equalize_adapthist(img, clip_limit=0.02)
        im_primary_width = img_primary.shape[1]
        plt.close('all')
        fig, ax = plt.subplots(figsize=(90, 30))
        real_annotation_frame = torch.zeros(4).to(torch.float64)
        virtual_annotation_frame = torch.zeros_like(real_annotation_frame).squeeze()
        for kpt_id in range(num_keypoints):
            # Show the image
            # Create a new figure and axis 
            ax.axis('off')                   
            
            cam_label = "primary_virtual"
            ax.imshow(img, cmap='gray')            
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                           

            # Get virtual annotations
            # Connect the click event to the onclick function
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            scat1 = None
            scat2 = None
            ax.set_title(f"Click on the virtual image: {image_file_primary} in primary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            #print(f"Click on the virtual image: {image_file_primary} in primary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            plt.show()
            len_coor = 0
            #NOTE: While loop for annotation of the primary camera
            while True:                
                if flag == 0:
                    flag = 1
                    break    
                plt.pause(0.0001)
            
                if len(coordinates) == len_coor:
                    continue
                # After closing the image, save the coordinates
                primary_virtual_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                coordinates.clear()
                
                # Compute the epipolar lines
                epipolar_line_unlabelled, epipolar_line_labelled = get_epipolar_line(arena, 
                                                primary_virtual_annotations_frame,
                                                cam_label=cam_label)  # cam_label is the label of the camera annotated by the user
                if cam_label == "primary_virtual":
                    epipolar_line_unlabelled[0, :] += im_primary_width 
                
                ax.imshow(img, cmap='gray')            
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                           

                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                    scat3.remove()
                scat1 = ax.scatter(
                    primary_virtual_annotations_frame[0,:].detach().numpy(),
                    primary_virtual_annotations_frame[1,:].detach().numpy(),
                    s=3,
                    color='g',
                )
                scat2 = ax.scatter(
                    epipolar_line_unlabelled[0,:].detach().numpy(),
                    epipolar_line_unlabelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.6
                )
                scat3 = ax.scatter(
                    epipolar_line_labelled[0,:].detach().numpy(),
                    epipolar_line_labelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.6
                )
                len_coor = len(coordinates)                
                plt.draw()
            
            
                # Compute the epipolar lines
                epipolar_line_unlabelled, epipolar_line_labelled = get_epipolar_line(arena, 
                                                primary_virtual_annotations_frame,
                                                cam_label=cam_label)  # cam_label is the label of the camera annotated by the user
                epipolar_line_unlabelled[0,:] += im_primary_width

            print(f'Clicked {coordinates}')
            cam_label = "primary_real"
            ax.imshow(img, cmap='gray')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                             
            ax.set_title(f"Click on the real image: {image_file} in primary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            #print((f"Click on the real image: {image_file} in primary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints"))
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            while True:                
                if flag == 0:
                    flag = 1
                    break    
                plt.pause(0.0001)
            
                if len(coordinates) == len_coor:
                    continue
                # After closing the image, save the coordinates
                primary_real_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                coordinates.clear()
                
                # Compute the epipolar lines
                epipolar_line_unlabelled = get_epipolar_line(arena, 
                                                primary_real_annotations_frame,
                                                cam_label=cam_label)  # cam_label is the label of the camera annotated by the user
                if cam_label == "primary_real":
                    epipolar_line_unlabelled[0, :] += im_primary_width 
                
                ax.imshow(img, cmap='gray')            
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                           

                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                    scat3.remove()
                scat1 = ax.scatter(
                    primary_real_annotations_frame[0,:].detach().numpy(),
                    primary_real_annotations_frame[1,:].detach().numpy(),
                    s=3,
                    color='g',
                )
                scat2 = ax.scatter(
                    epipolar_line_unlabelled[0,:].detach().numpy(),
                    epipolar_line_unlabelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.75
                )
                # Plot from previous virtual annotation in primary camera
                scat3 = ax.scatter(
                    epipolar_line_labelled[0,:].detach().numpy(),
                    epipolar_line_labelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.6
                )
                len_coor = len(coordinates)                
                plt.draw()
            
                # Compute the epipolar lines
                epipolar_line_unlabelled = get_epipolar_line(arena, 
                                                primary_real_annotations_frame,
                                                cam_label="primary_real")  # cam_label is the label of the camera annotated by the user
                epipolar_line_unlabelled[0,:] += im_primary_width


            cam_label = 'secondary_virtual'
            ax.imshow(img, cmap='gray')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                             
            ax.set_title(f"Click on the virtual image: {image_file} in secondary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            #print(f"Click on the virtual image: {image_file} in secondary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            cid = fig.canvas.mpl_connect('button_press_event', onclick)            
            #NOTE: While loop for annotation of the secondary camera
            while True:
                if flag == 0:
                    flag = 1
                    break
                plt.pause(0.0001)
                
                if len_coor == len(coordinates):
                    continue

                secondary_virtual_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                secondary_virtual_annotations_frame[0,:] -= im_primary_width # Accounting for the stacking of primary and secondary image
                coordinates.clear()
                
                
                epipolar_line_unlabelled, epipolar_line_labelled = get_epipolar_line(arena, 
                                                  secondary_virtual_annotations_frame,
                                                  cam_label=cam_label)  # cam_label is the label of the camera annotated by the user
                

                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                    scat3.remove()

                ax.imshow(img, cmap='gray')            
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                           

                scat1 = ax.scatter(
                        epipolar_line_labelled[0,:].detach().numpy() + im_primary_width,
                        epipolar_line_labelled[1,:].detach().numpy(),
                        s=0.1,
                        color='r',
                        alpha=0.6
                    )
                
                scat2 = ax.scatter(
                        epipolar_line_unlabelled[0,:].detach().numpy(),
                        epipolar_line_unlabelled[1,:].detach().numpy(),
                        s=0.1,
                        color='r',
                        alpha=0.6
                    )
                scat3 = ax.scatter(
                        secondary_virtual_annotations_frame[0,:].detach().numpy() + im_primary_width,
                        secondary_virtual_annotations_frame[1,:].detach().numpy(), # Accounting for the stacking of primary and secondary image
                        s=2,
                        color='g',
                    )
                len_coor = len(coordinates)
                plt.draw()

            cam_label = "secondary_real"
            ax.imshow(img, cmap='gray')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                             
            ax.set_title(f"Click on the real image: {image_file} in secondary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            #print(f"Click on the real image: {image_file} in secondary camera to annotate {keypoints_dict[kpt_id]}, {kpt_id+1} of {num_keypoints} keypoints")
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            while True:                
                if flag == 0:
                    flag = 1
                    break
                plt.pause(0.0001)
            
                if len(coordinates) == len_coor:
                    continue
                # After closing the image, save the coordinates
                secondary_real_annotations_frame = torch.tensor(coordinates[-1]).to(torch.float64)[:,None]
                secondary_real_annotations_frame[0, :] -= im_primary_width # Accounting for the stacking of primary and secondary image
                coordinates.clear()
                
                # Compute the epipolar lines
                epipolar_line_unlabelled = get_epipolar_line(arena, 
                                                secondary_real_annotations_frame,
                                                cam_label=cam_label)  # cam_label is the label of the camera annotated by the user
                if cam_label == "primary_real":
                    epipolar_line_unlabelled[0, :] += im_primary_width 
                
                ax.imshow(img, cmap='gray')            
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)                           

                if scat1 is not None:
                    scat1.remove()
                    scat2.remove()
                    scat3.remove()
                scat1 = ax.scatter(
                    secondary_real_annotations_frame[0,:].detach().numpy() + im_primary_width, 
                    secondary_real_annotations_frame[1,:].detach().numpy(),
                    s=3,
                    color='g',
                )
                scat2 = ax.scatter(
                    epipolar_line_unlabelled[0,:].detach().numpy(),
                    epipolar_line_unlabelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.75
                )
                # Plot from previous virtual annotation in primary camera
                scat3 = ax.scatter(
                    epipolar_line_labelled[0,:].detach().numpy(),
                    epipolar_line_labelled[1,:].detach().numpy(),
                    s=0.1,
                    color='r',
                    alpha=0.6
                )
                len_coor = len(coordinates)                
                plt.draw()
            
            
                # Compute the epipolar lines
                epipolar_line_unlabelled = get_epipolar_line(
                                            arena, 
                                                primary_virtual_annotations_frame,
                                                cam_label="secondary_real")  # cam_label is the label of the camera annotated by the user

            # Clear the coordinates for the next image
            coordinates.clear()
            

            #NOTE: Annotation for the current frame ends here
            image_path = image_path_primary.replace('/cam_0', '')
            real_annotation_frame[:2] = primary_real_annotations_frame.squeeze()
            real_annotation_frame[2:] = secondary_real_annotations_frame.squeeze()
            
            virtual_annotation_frame[:2] = primary_virtual_annotations_frame.squeeze()
            virtual_annotation_frame[2:] = secondary_virtual_annotations_frame.squeeze()
            print(f'virtual annotation frame {virtual_annotation_frame}')
            print(f'primary virtual annotations {primary_virtual_annotations_frame}')
        
            annotations = convert_annotations_to_excel(real_annotation_frame, virtual_annotation_frame, im_id=image_path, kpt_id=kpt_id)
       
            if excel_file_path is not None:
                append_to_excel(excel_file_path, annotations, keypoints_dict)
            else:
                print("Excel file path not provided. Annotations not saved.")
            primary_virtual_annotations[image_id, :, kpt_id] = primary_virtual_annotations_frame[:,0]
            secondary_virtual_annotations[image_id, :, kpt_id] = secondary_virtual_annotations_frame[:,0]
            primary_real_annotations[image_id, :, kpt_id] = primary_real_annotations_frame[:,0]
            secondary_real_annotations[image_id, :, kpt_id] = secondary_real_annotations_frame[:,0]

            flag = 1
            ax.cla()
    
    return primary_virtual_annotations, secondary_virtual_annotations, primary_real_annotations, secondary_real_annotations

# %%
def get_secondary_camera(arena):
    """
    Returns secondary camera given the arena model parameters
    """
    R_stereo_cam = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            )
    return arena.get_stereo_camera(arena.principal_point_pixel_cam_1,
                                arena.focal_length_cam_1,
                                R_stereo_cam,
                                arena.T_stereo_cam,
                                r1=arena.stereocam_r1,
                                radial_dist_coeffs=arena.radial_dist_coeffs_cam_1)
    


# %%
excel_file_name= f'{images_dir}/annotations.xlsx'
keypoints_dict = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'Head', 'Belly', 'Thorax Tip', 'Head Tip', 'l1', 'r1', 'l2', 'r2', 'l3', 'r3']
keypoints_dict = ['L1', 'R1', 'Neck']
excel_file_path = os.path.join(experiment_dir, excel_file_name)
virtual_annotations, real_annotations = get_user_annotations(arena, images_dir, keypoints_dict, first_frame=1,
excel_file_path=excel_file_path, step=2)