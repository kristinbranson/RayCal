from scipy.optimize import least_squares
import time
import numpy as np
import matplotlib.pyplot as plt

def rescale_pixels(p_scaled, focal_length, principal_point):
     return p_scaled * focal_length + principal_point


def scale_pixels(p_unscaled, focal_length, principal_point):
     return (p_unscaled - principal_point) / focal_length

num_pixels = 1200
focal_length = 6000
principal_point = np.reshape(np.array([640, 640]), (2,1))
distortion_params = np.array([10, 10])
pixels_undistorted_unscaled = 1280 * (np.random.rand(2, num_pixels))
pixels_undistorted = scale_pixels(pixels_undistorted_unscaled, focal_length, principal_point)
r2 = pixels_undistorted[0,:] ** 2 + pixels_undistorted[1,:] ** 2
r4 = r2**2

p_distorted = np.zeros_like(pixels_undistorted).copy()
p_distorted[0,:] = pixels_undistorted[0,:].copy() * (1 + distortion_params[0] * r2 - distortion_params[1] * r4)
p_distorted[1,:] = pixels_undistorted[1,:].copy() * (1 + distortion_params[0] * r2 - distortion_params[1] * r4)
pixels_distorted_unscaled = rescale_pixels(p_distorted, focal_length, principal_point)


def cost_func_params(x, p_undistorted, p_output):
	r2 = p_undistorted[0,:] ** 2 + p_undistorted[1,:] ** 2
	r4 = r2 ** 2
	p_predict = np.zeros_like(p_undistorted)
	p_predict[0,:] = p_undistorted[0,:] * (1 + x[0] * r2 + x[1] * r4)
	p_predict[1,:] = p_undistorted[1,:] * (1 + x[0] * r2 + x[1] * r4)
	return np.linalg.norm(p_predict - p_output, axis=0)


def cost_func_distort(p_undistorted, p_distorted, dist_params_):
	r2 = p_undistorted[0,:] ** 2 + p_undistorted[1,:] ** 2
	r4 = r2 ** 2
	p_predict = np.zeros_like(p_undistorted)
	p_predict[0,:] = p_undistorted[0,:] * (1 + dist_params_[0] * r2 + dist_params_[1] * r4)
	p_predict[1,:] = p_undistorted[1,:] * (1 + dist_params_[0] * r2 + dist_params_[1] * r4)
	return np.linalg.norm(p_predict - p_distorted, axis=0)


def cost_func_distort_flatten(p_undistorted_flat, p_distorted, dist_params_):
    num_pixels = p_distorted.shape[1]
    p_undistorted = np.reshape(p_undistorted_flat, (2, num_pixels))
    r2 = (p_undistorted[0,:] ** 2 + p_undistorted[1,:] ** 2)[None, :]
    r4 = r2 ** 2
    pixels_distorted_pred = pixels_undistorted * (1 + dist_params_[0] * r2 + dist_params_[1] * r4)
    return np.linalg.norm(pixels_distorted_pred - p_distorted, axis=0)


def fit_distortion_parameters(pixels_undistorted, pixels_distorted):
    coeffs_list = least_squares(fun=cost_func_params, x0=np.array([0.,-0.]), args=([pixels_undistorted, pixels_distorted]))
    return coeffs_list[-1]

def fit_distortion_parameters_flatten(pixels_distorted, distortion_params):
    p_init = pixels_distorted.flatten()
    coeffs_list = least_squares(fun=cost_func_distort_flatten, x0=p_init, args=([pixels_distorted, distortion_params]))
    return coeffs_list

def estimate_undistorted_pixels(pixels_distorted, distortion_params_):
    max_iterations = 100
    tolerance = 1e-10
    pixels_undistorted = pixels_distorted.copy()
    for i in range(max_iterations):
        # Calculate the radial distance squared
        r2 = pixels_undistorted[0, :] ** 2 + pixels_undistorted[1, :] ** 2
        r4 = r2 ** 2
        # Calculate the radial distortion factor
        radial_distortion = 1 + distortion_params_[0] * r2 + distortion_params_[1] * r4
        # Update undistorted coordinates
        pixels_undistorted_new = pixels_distorted / radial_distortion
        # Check for convergence
        if np.max(np.abs(pixels_undistorted_new - pixels_undistorted)) < tolerance:
            print(f'Converged at iteration {i}')
            break
        # Update for the next iteration
        pixels_undistorted = pixels_undistorted_new
    return pixels_undistorted


if __name__ == "__main__":
    start_time = time.time()
    """
    coeffs_list = fit_distortion_parameters_flatten(p_distorted, distortion_params)
    print(f'Time elapsed: {time.time() - start_time}')
    recon = coeffs_list.x
    recon = np.reshape(recon, (2,num_pixels))
    """

    # Fixed point convergence
    recon = estimate_undistorted_pixels(p_distorted, distortion_params)
    print(f'Convergence test: {np.linalg.norm(pixels_undistorted - recon, axis=0).mean()}')

    # Plot
    recon = rescale_pixels(recon, focal_length=focal_length, principal_point=principal_point)
    plt.scatter(recon[0,:],
                recon[1,:])
    plt.scatter(pixels_undistorted_unscaled[0,:],
                pixels_undistorted_unscaled[1,:],
                marker='x')

