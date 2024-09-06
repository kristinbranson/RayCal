# %% Imports
from ray_tracing_simulator import Prism, Ray, Plane, OpticalPlane, visualize_camera_configuration
import matplotlib.pyplot as plt
import numpy as np  
import torch
pi = torch.tensor(np.pi)

# %% Testing class Plane
# Single plane
plane = Plane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[1.,0.,0.], a=1.,b=1.)
_, ax = plane.visualize()
ax.set_aspect('equal', adjustable='datalim')   


# %% Testing class Ray
# Single ray
ray = Ray(origin=[0.,0.,0.], direction=[1.,0.,0.])
_, ax = ray.visualize()
ax.set_aspect('equal', adjustable='datalim')   

# %% Testing class OpticalPlane with a single ray refraction 
# Optical plane refraction
plane = OpticalPlane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[0.,0.,0.], a=1.,b=1., 
              refractive_idx_1=1., refractive_idx_2=1.5)
incident_ray = Ray(origin=[-0.5,0.,0.], direction=[1.,0.,0.])
refracted_ray = plane.refract_ray(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = refracted_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 

# %% Testing class OpticalPlane with a single ray reflection
# Optical plane reflection
plane = OpticalPlane(alpha=np.pi/3, beta=np.pi/6, gamma=np.pi/10, center=[0.,0.,0.], a=1.,b=1., 
              refractive_idx_1=1., refractive_idx_2=1.5)
incident_ray = Ray(origin=[-0.5,0.,0.], direction=[1.,0.,0.])
reflected_ray = plane.reflect_ray(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = reflected_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 

# %% Testing class Prism 
# Testing a single ray refracting and reflecting through a prism
n_glass = 1.55
n_air = 1.
prism_alpha = 0.
prism_beta = np.pi / 2
prism_gamma = np.pi / 2 
prism_center = np.array([0.,0.,0.])[:, None]

prism = Prism(prism_size=[1.,1.,1.], prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                prism_center=prism_center, refractive_index_glass=n_glass, 
                refractive_index_air=n_air)

origin_point=[0.,0.6,-0.25] 
target_point=[0.,0.37,0.]
prism.trace_ray(origin_point, target_point) 
fig, ax = prism.visualize_prism_and_ray()
ax.set_aspect('equal', adjustable='datalim')        
plt.show()

# %% Test camera configuration
pixels = torch.rand(2, 10)
pixels = pixels * torch.tensor([1918, 1200])[:, None]
_, _, prism, camera = visualize_camera_configuration(pixels=pixels)
# %%
