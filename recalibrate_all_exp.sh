#bsub -n 8 -J "recalibrate" -o recalibrate.log<< 'EOF'
#!/bin/bash
conda activate
source /groups/branson/bransonlab/aniket/pytorch_remote/bin/activate

exp_data_dir='/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led'
for dir in ${exp_data_dir}/exp_*; do
    # Check if directory exists (handles case where no exp_* directories exist)
    if [ ! -d "$dir" ]; then
	echo "Did not find $dir"
        continue
    fi
    
    # Check if the calibration file exists in this directory
    calibration_image_0=($dir/calibration_grid_images/initialization_cam_0.*)
    calibration_image_1=($dir/calibration_grid_images/initialization_cam_1.*)
    
    
    if [ -f "$calibration_image_0" ] && [ -f "$calibration_image_1" ]; then
        # Remove trailing slash for cleaner path
        clean_dir="${dir%/}"
        
	exp_name=$(basename $dir)
        echo "Processing directory: $exp_name"
	bsub -n 6 -J recalibrate_${exp_name} -o logs/recalibrate_${exp_name}.log python runme_train_simulator_camera_prism_grid_distances.py --exp_id $exp_name
    else
	echo "Did not find: $dir"
    fi
done
#EOF
