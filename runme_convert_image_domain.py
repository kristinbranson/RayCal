import matplotlib.pyplot as plt
import os
import numpy as np

fig = plt.figure(figsize=(25, 15))


image_folder = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_13/fly_images/cam_0/image_cam_0_date_2024_12_17_time_21_03_30_v001_selected'

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', 'bmp'))]
image_file = image_files[0]
image_path = os.path.join(image_folder, image_file)

offsets = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
# Add six subplots
for i in range(8):
    original_image = plt.imread(image_path)
    image = offsets[i] - original_image
    image = np.clip(image, 0, 1)
    image = image + offsets[i]

    ax = fig.add_subplot(2, 4, i+1)
    if i == 0:
        image = original_image
    ax.imshow(image, cmap='gray')
    if i == 0:
        ax.set_title('Original Image')
    else:
        ax.set_title(f'Offset {offsets[i]}')
    # Add border around subplot
    for spine in ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)