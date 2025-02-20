from torchimize.functions import lsq_lma
import torch
import pdb
import time
import matplotlib.pyplot as plt
import numpy as np

def rescale_pixels(p_scaled, focal_length, principal_point):
     return p_scaled * focal_length + principal_point


def scale_pixels(p_unscaled, focal_length, principal_point):
     return (p_unscaled - principal_point) / focal_length

focal_length = 6000
principal_point = torch.tensor([640, 640]).unsqueeze(-1)
num_pixels = 1028
distortion_params = torch.tensor([-10, 30])

pixels_undistorted_unscaled = 1200 * (torch.rand(2, num_pixels)).to(torch.float64)
pixels_undistorted = scale_pixels(pixels_undistorted_unscaled, focal_length, principal_point)
r2 = pixels_undistorted[0,:] ** 2 + pixels_undistorted[1,:] ** 2
r4 = r2**2

pixels_distorted = torch.zeros_like(pixels_undistorted).clone()
pixels_distorted[0,:] = pixels_undistorted[0,:].clone() * (1 + distortion_params[0] * r2 + distortion_params[1] * r4)
pixels_distorted[1,:] = pixels_undistorted[1,:].clone() * (1 + distortion_params[0] * r2 + distortion_params[1] * r4)
pixels_distorted_unscaled = scale_pixels(pixels_distorted, focal_length, principal_point)


def cost_func(x, p_input, p_output):
	r2 = p_input[0,:] ** 2 + p_input[1,:] ** 2
	r4 = r2 ** 2
	p_predict = torch.zeros_like(p_input)
	p_predict[0,:] = p_input[0,:] * (1 + x[0] * r2 + x[1] * r4)
	p_predict[1,:] = p_input[1,:] * (1 + x[0] * r2 + x[1] * r4)
	return torch.norm(p_predict - p_output, dim=0)


def fit_distortion_parameters(pixels_undistorted, pixels_distorted):
    coeffs_list = lsq_lma(torch.tensor([0.,0.]).to(torch.float64), function=cost_func, args=(pixels_undistorted, pixels_distorted))
    return coeffs_list[-1]


def estimate_undistorted_pixels(pixels_distorted, distortion_params_):
    max_iterations = 10000
    tolerance = 1e-11
    #distortion_params_ = distortion_params_.numpy()
    #pixels_distorted = pixels_distorted.numpy()
    pixels_undistorted = pixels_distorted.clone()
    for i in range(max_iterations):
        # Calculate the radial distance squared
        r2 = pixels_undistorted[0, :] ** 2 + pixels_undistorted[1, :] ** 2
        #r2 = torch.norm(pixels_undistorted, dim=0) ** 2
        r4 = r2 ** 2
        # Calculate the radial distortion factor
        radial_distortion = 1 + distortion_params_[0] * r2 + distortion_params_[1] * r4
        # Update undistorted coordinates
        pixels_undistorted_new = pixels_distorted / radial_distortion
        # Check for convergence
        if torch.max(torch.abs(pixels_undistorted_new - pixels_undistorted)) < tolerance:
            print(f'Converged at iteration {i}')
            break
        # Update for the next iteration
        pixels_undistorted = pixels_undistorted_new
    return pixels_undistorted


if __name__ == "__main__":
    start_time = time.time()
    # Levenberg Marquadt
    #coeffs_list = lsq_lma(torch.tensor([0.,0.]).to(torch.float64), function=cost_func, args=(pixels_undistorted, pixels_distorted))
    
    
    # Fixed point convergence
    num_tests = 100
    for _ in range(num_tests):
        recon = estimate_undistorted_pixels(pixels_distorted, distortion_params)
    print(f'Average time per optimization: {(time.time() - start_time) / num_tests}')

    # Plot
    recon = rescale_pixels(recon, focal_length=focal_length, principal_point=principal_point)
    print(f'Convergence test: {torch.norm(pixels_undistorted_unscaled - recon, dim=0).mean()}')
    plt.scatter(recon[0,:],
                recon[1,:],
                label='Reconstructed',
                color='r')
    plt.scatter(pixels_undistorted_unscaled[0,:],
                pixels_undistorted_unscaled[1,:],
                marker='x',
                label='Original',
                color='g')
    plt.legend()
    