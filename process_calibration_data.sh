# Template for this code was produced using GPT-4o mini
#!/bin/bash

rootDataDir='/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/' # This is the directory where imaging data from all experiments is stored in separate folders named exp_xx
# Check if a directory argument is provided
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <experiment_id (int)>"
    exit 1
fi

# Specify the directory to check from the input argument
experimentDataDir="$rootDataDir/exp_$1/calibration_grid_images/"

# Check if the provided argument is a valid directory
if [ ! -d "$experimentDataDir" ]; then
    echo "Error: '$experimentDataDir' is not a valid directory."
    exit 1
fi

cam0Dir="$experimentDataDir/cam_0"
cam1Dir="$experimentDataDir/cam_1"
cam02Dir="$experimentDataDir/cam_02"
cam13Dir="$experimentDataDir/cam_13"

#Count the number of directories in the specified directory
FOLDER_COUNT0=$(find "$cam0Dir" -maxdepth 1 -type d | wc -l)
# Subtract 1 from the count to exclude the current directory (.)
FOLDER_COUNT0=$((FOLDER_COUNT0 - 1))

# Count the number of directories in the specified directory
FOLDER_COUNT1=$(find "$cam1Dir" -maxdepth 1 -type d | wc -l)
# Subtract 1 from the count to exclude the current directory (.)
FOLDER_COUNT1=$((FOLDER_COUNT1 - 1))

# Count the number of directories in the specified directory
FOLDER_COUNT02=$(find "$cam02Dir" -maxdepth 1 -type d | wc -l)
# Subtract 1 from the count to exclude the current directory (.)
FOLDER_COUNT02=$((FOLDER_COUNT02 - 1))

# Count the number of directories in the specified directory
FOLDER_COUNT13=$(find "$cam13Dir" -maxdepth 1 -type d | wc -l)
# Subtract 1 from the count to exclude the current directory (.)
FOLDER_COUNT13=$((FOLDER_COUNT13 - 1))

echo "ASDASD"

# Check if the number of folders is greater than 1
if [ "$FOLDER_COUNT0" -gt 1 ] || [ "$FOLDER_COUNT0" -lt 1 ]; then
    echo "Exiting: $cam0Dir should have only 1 folder but $FOLDER_COUNT0 were found"
    exit 1
else
    echo "$cam0Dir has only one folder. Check passed."
fi

if [ "$FOLDER_COUNT1" -gt 1 ] || [ "$FOLDER_COUNT1" -lt 1 ]; then
    echo "Exiting: $cam1Dir should have only 1 folder but $FOLDER_COUNT1 were found"
    exit 1
else
    echo "$cam1Dir has only one folder. Check passed."
fi

if [ "$FOLDER_COUNT02" -gt 1 ] || [ "$FOLDER_COUNT02" -lt 1 ]; then
    echo "Exiting: $cam02Dir should have only 1 folder but $FOLDER_COUNT02 were found"
    exit 1
else
    echo "$cam0Dir has only one folder. Check passed."
fi

if [ "$FOLDER_COUNT13" -gt 1 ] || [ "$FOLDER_COUNT13" -lt 1 ]; then
    echo "Exiting: $cam13Dir should have only 1 folder but $FOLDER_COUNT13 were found"
    exit 1
else
    echo "$cam13Dir has only one folder. Check passed."
fi

# List the folders
imageDir=$(find "$cam0Dir" -maxdepth 1 -type d -name "image_*")
mjpgPath=$(find "$imageDir" -type f -name '*.mjpg')
echo "$imageDir"
ffmpeg -i "$mjpgPath" "$cam0Dir/image_%04d.png"

imageDir=$(find "$cam1Dir" -maxdepth 1 -type d -name "image_*")
mjpgPath=$(find "$imageDir" -type f -name '*.mjpg')
ffmpeg -i "$mjpgPath" "$cam1Dir/image_%04d.png"

imageDir=$(find "$cam02Dir" -maxdepth 1 -type d -name "image_*")
mjpgPath=$(find "$imageDir" -type f -name '*.mjpg')
ffmpeg -i "$mjpgPath" "$cam02Dir/image_%04d.png"

imageDir=$(find "$cam13Dir" -maxdepth 1 -type d -name "image_*")
mjpgPath=$(find "$imageDir" -type f -name '*.mjpg')
ffmpeg -i "$mjpgPath" "$cam13Dir/image_%04d.png"

echo "Detecting and saving dotted grids from cam_0 and cam_1. These will be used for estimating camera intrinsics"
/misc/local/matlab-2023b/bin/matlab -batch "exp_id = $1; run('matlab_scripts/runme_annotate_circular_grid_points_automated.m')"

echo "Calibrating camera intrinsics and saving them"
/misc/local/matlab-2023b/bin/matlab -batch "exp_id = $1; run('matlab_scripts/runme_calibrate_grid_automated.m')"

echo "Detecting and saving dotted grids from cam_02 and cam_13"
/misc/local/matlab-2023b/bin/matlab -batch "exp_id = $1; run('matlab_scripts/runme_annotate_grid_prism.m')"

echo "Exporting grid coordinates in a format ready for calibration"
/misc/local/matlab-2023b/bin/matlab -batch "exp_id = $1; run('matlab_scripts/runme_export_data_two_cams.m')"

echo "Exporting prism initialization"
/misc/local/matlab-2023b/bin/matlab -batch "exp_id = $1; run('matlab_scripts/runme_annotate_prism_initialization_image.m')"

echo "Training calibration model using pytorch-based ray-tracing"
python runme_train_simulator_camera_prism_grid_distances.py --exp_id $1
