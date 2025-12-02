# Template for this code was produced using GPT-4o mini
#!/bin/bash
source config.log

# Check if a directory argument is provided
if [ "$#" -lt 1 ]; then
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

flyDircam0="$rootDataDir/exp_$1/fly_images/cam_0/"
flyDircam1="$rootDataDir/exp_$1/fly_images/cam_1/"
flyDircam0Cropped="$rootDataDir/exp_$1/fly_images/cropped_uniform_sizes"
flyDircam1Cropped="$rootDataDir/exp_$1/fly_images/cropped_uniform_sizes"
echo "Creating directories to save cropped .ufmf fly videos $flyDircam0Cropped and $flyDircam1Cropped"
mkdir $flyDircam0Cropped
mkdir $flyDircam1Cropped

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
# Unpack all video files 

if [ "$2" -ne 0 ]; then
    echo "Unpacking mjpg files"
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
else
    echo "Skipping unpacking mjpg files"
fi

#Annotating dividing column
echo "Estimating grid coordinates"
/misc/local/matlab-2023a/bin/matlab -r "exp_id = $1; dataDir = '$rootDataDir'; run('matlab_scripts/runme_annotate_dividing_column.m'); run('matlab_scripts/runme_annotate_circular_grid_points_automated.m'); run('matlab_scripts/runme_calibrate_grid_automated.m'); run('matlab_scripts/runme_annotate_grid_prism.m'); run('matlab_scripts/runme_export_data_two_cams.m'); run('matlab_scripts/runme_annotate_prism_initialization_image.m'); exit()"

echo "Finished estimating grid coordinates"
echo "Exported dividing columns"
cal_grid_log_path="$rootDataDir/exp_$1/calibration_grid_images/logfile.txt"
echo "Temporary saving log data for the two camera images in $cal_grid_log_path"
mapfile -t log < $cal_grid_log_path
dividing_col=(${log[@]:0:2})
image_width=(${log[@]: -2})

# Print the integers
echo "Dividing column for the first image: ${dividing_col[0]}"
echo "Image width for the first image: ${image_width[0]}"
echo "Dividing column for the second image: ${dividing_col[1]}"
echo "Image width for the second image: ${image_width[1]}"

echo "Training calibration model using pytorch-based ray-tracing"

export PYTHONPATH="$APT_path/deepnet/"

# Compute cropping parameters and crop movies to ufmf file: each movie frame from raw ufmf files is converted into two frames comprising the virtual and real view
starting_col=(0 0)
crop_lower_real_view=(0 0) #Lower cropping bound for the real view for either of the cameras
crop_upper_real_view=(0 0) #Upper cropping bound for the real view for either of the cameras

if [ ${dividing_col[0]} -gt $virtual_view_size ]; then
    starting_col[0]=$((dividing_col[0] - virtual_view_size))
else
    starting_col[0]=0
    dividing_col[0]=$virtual_view_size
fi

if [ ${dividing_col[1]} -gt $virtual_view_size ]; then
    starting_col[1]=$((dividing_col[1] - virtual_view_size))
else
    starting_col[1]=0
    dividing_col[1]=$virtual_view_size
fi

if [ $[image_width[0]-dividing_col[0]] -gt $real_view_size ]; then
	crop_lower_real_view[0]=${dividing_col[0]}
	crop_upper_real_view[0]=$((dividing_col[0] + real_view_size))
else
	crop_lower_real_view[0]=$((image_width[0] - real_view_size))
	crop_upper_real_view[0]=$[image_width[0] - 1]
fi

if [ $[image_width[1]-dividing_col[1]] -gt $real_view_size ]; then
        crop_lower_real_view[1]=${dividing_col[1]}
        crop_upper_real_view[1]=$((dividing_col[1] + real_view_size))
else
        crop_lower_real_view[1]=$((image_width[1] - real_view_size))
        crop_upper_real_view[1]=$[image_width[1] - 1]
fi

# Pick the more generous cropping parameteres of the two cameras and assign them consistently across botht the cameras
if [[ ${dividing_col[0]} -gt ${dividing_col[1]} ]]; then
    dividing_col[1]=${dividing_col[0]}
else
    dividing_col[0]=${dividing_col[1]}
fi

if [[ ${starting_col[0]} -gt ${starting_col[1]} ]]; then
        starting_col[0]=${starting_col[1]}
else
        starting_col[1]=${starting_col[0]}
fi

if [[ ${crop_upper_real_view[0]} -gt ${crop_upper_real_view[1]} ]]; then
	crop_upper_real_view[1]=${crop_upper_real_view[0]}
else
	crop_upper_real_view[0]=${crop_upper_real_view[1]}
fi

if [[ ${crop_lower_real_view[0]} -gt ${crop_lower_real_view[1]} ]]; then
        crop_lower_real_view[0]=${crop_lower_real_view[1]}
else
        crop_lower_real_view[1]=${crop_lower_real_view[0]}
fi

echo "Cropping parameters for cam_0 (secondary camera)"
echo "crop cols: [[${starting_col[0]},${dividing_col[0]}],[${crop_lower_real_view[0]},${crop_upper_real_view[0]}]]"
echo "Cropping parameters for cam_1 (primary camera)"
echo "crop cols: [[${starting_col[1]},${dividing_col[1]}],[${crop_lower_real_view[1]},${crop_upper_real_view[1]}]]"


# Rotate, crop and save ufmf for cam_0 (primary camera)
find "$flyDircam0" -maxdepth 1 -type f -name "image_*" | while IFS= read -r file; do
    echo "Processing file: $file"
    echo "crop cols: [[${starting_col[0]},${dividing_col[0]}],[${crop_lower_real_view[0]},${crop_upper_real_view[0]}]]"
    python /groups/branson/bransonlab/aniket/APT/deepnet/crop_ufmf.py $file --croprows "[[0,-1],[0,-1]]" --cropcols "[[${starting_col[0]},${dividing_col[0]}],[$crop_lower_real_view,$crop_upper_real_view]]" --outdir $flyDircam0Cropped/ --rot90 1
done

# Rotate, crop and save ufmf for cam_1 (secondary camera)
find "$flyDircam1" -maxdepth 1 -type f -name "image_*" | while IFS= read -r file; do
    echo "Processing file: $file"
    echo "crop cols: [[${starting_col[1]},${dividing_col[1]}],[$crop_lower_real_view[1],$crop_upper_real_view[1]]]"
    python /groups/branson/bransonlab/aniket/APT/deepnet/crop_ufmf.py $file --croprows "[[0,-1],[0,-1]]" --cropcols "[[${starting_col[1]},${dividing_col[1]}],[${crop_lower_real_view[1]},${crop_upper_real_view[1]}]]" --outdir $flyDircam1Cropped/ --rot90 1
done


# Write useful cropping parameters to a temporary YAML file 'temp.yaml'
yaml_file="temp.yaml"
echo "dividing_col: [${dividing_col[0]}, ${dividing_col[1]}]" > "$yaml_file"
echo "crop_upper_real_view: [${crop_upper_real_view[0]}, ${crop_upper_real_view[1]}]" > "$yaml_file"
echo "image_width: [${image_width[0]}, ${image_width[1]}]">> "$yaml_file"

#TODO: Was the following line necessary
#echo "python_script: '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/pyCall/return_projected_ray_two_cameras_prism.py'" >> "$yaml_file"

python runme_train_simulator_camera_prism_grid_distances.py --exp_id $1
model_file_name='best_model_weights_only.pth'

#read -r dividing_col image_width python_script_path model_path<<< $(python - <<EOF
eval "$(python - <<END

import yaml

import yaml

with open("$yaml_file", 'r') as file:
    data = yaml.safe_load(file)

print('dividing_col=(' + ' '.join(map(str, dividing_col)) + ')')
print('image_width=(' + ' '.join(map(str, image_width)) + ')')
python_script_path=data['python_script']
model_path=data['model_path']
print(f"{dividing_col} {image_width} {python_script_path} {model_path}")
END
)"


#mkdir "$root_data_directory/exp_$1/fly_images/cropped_uniform_sizes"
calibration_files_path = "$root_data_directory/exp_$1/calibration_files"
mkdir $calibration_files_path
#mkdir "$PTR_dir/exp$1/"
#mkdir "$PTR_dir/exp$1/movies"

# Save calibration.mat file by running a MATLAB instance
/misc/local/matlab-2023a/bin/matlab -batch "calibrations = []; clipping_col=[${crop_upper_real_view[0]}, ${crop_upper_real_view[1]}]; dividing_col=[${dividing_col[0]}, ${dividing_col[1]}]; image_width=[${image_width[0]}, ${image_width[1]}]; model_path='$PTR_dir/exp$1/$model_file_name'; nviews=4; python_script_path='$python_script_path'; raytracing=1; save(['$calibration_files_path']); exit;"


#cp $APT_path/calibration_data/exp_$1_calibration_data.mat $PTR_dir/exp$1/
# Copy .pth file
cp $model_path $calibration_files_path


#cp $flyDircam0Cropped/*.ufmf $PTR_dir/exp$1/movies
#cp $flyDircam1Cropped/*.ufmf $PTR_dir/exp$1/movies
#cp $model_path $PTR_dir/exp$1/
