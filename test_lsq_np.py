from scipy.optimize import least_squares
import time
import numpy as np

pixels = 0.5 * (np.random.rand(2, 500) - 0.5)
r2 = pixels[0,:] ** 2 + pixels[1,:] ** 2
r4 = r2**2

p_ = np.zeros_like(pixels).copy()
p_[0,:] = pixels[0,:].copy() * (1 + 2 * r2 - 10 * r4)
p_[1,:] = pixels[1,:].copy() * (1 + 2 * r2 - 10 * r4)

def cost_func(x, p, p_):
	r2 = p[0,:] ** 2 + p[1,:] ** 2
	r4 = r2 ** 2
	p_predict = np.zeros_like(p)
	p_predict[0,:] = p[0,:] * (1 + x[0] * r2 + x[1] * r4)
	p_predict[1,:] = p[1,:] * (1 + x[0] * r2 + x[1] * r4)
	return np.linalg.norm(p_predict - p_, axis=0)

start_time = time.time()
coeffs_list = least_squares(fun=cost_func, x0=np.array([0.,-0.]), args=([pixels, p_]))
print(f'Time elapsed: {time.time() - start_time}')
print(f'Solution: {coeffs_list.x}')
