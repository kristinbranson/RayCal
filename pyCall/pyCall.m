pyenv('Version','/groups/branson/bransonlab/aniket/pytorch_remote/bin/python')
epipolar_line = pyrunfile("return_projected_ray.py", "epipolar_line", user_annotation=[400,500]);
epipolar_line = double(epipolar_line);