from torchimize.functions import lsq_lma
import torch
import pdb
import time

pixels = 0.5 * (torch.rand(2, 500) - 0.5).to(torch.float64)
r2 = pixels[0,:] ** 2 + pixels[1,:] ** 2
r4 = r2**2

p_ = torch.zeros_like(pixels).clone()
p_[0,:] = pixels[0,:].clone() * (1 + 10 * r2 - 20 * r4)
p_[1,:] = pixels[1,:].clone() * (1 + 10 * r2 - 20 * r4)
def cost_func(x, p, p_):
	r2 = p[0,:] ** 2 + p[1,:] ** 2
	r4 = r2 ** 2
	p_predict = torch.zeros_like(p)
	p_predict[0,:] = p[0,:] * (1 + x[0] * r2 + x[1] * r4)
	p_predict[1,:] = p[1,:] * (1 + x[0] * r2 + x[1] * r4)
	return torch.norm(p_predict - p_, dim=0)

start_time = time.time()
coeffs_list = lsq_lma(torch.tensor([0.,0.]).to(torch.float64), function=cost_func, args=(pixels, p_))
print(time.time() - start_time)
print(coeffs_list[-1])
#print(cost_func(coeffs_list[-1].sum().item(), pixels, p_))
