import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from config import Config
mpl.use('tkAgg')
plt.ion()

# %% parameters

# number of points to simulate
npts = Config.num_points

# angle of camera
theta_camera = Config.theta_camera

# camera focal length
focal_length = Config.focal_length

# camera image limits
minim = -Config.image_height
maxim = 0
# image are all at x = -focal_length, y from minim to maxim

nair = Config.refractive_index_air
nglass = Config.refractive_index_glass

# distance from the camera to the first face
distface1 = Config.prism_camera_distance

# distance along diagonal from the last ray to the bottom of the prism
distprismbottom = 5
# distance along diagonal from the first ray to the top of the prism
distprismtop = 2

# for refractive only case, thickness of the glass
distface2 = Config.prism_thickness

# distance from the planar object to the last face
distfinal = Config.prism_object_distance

# %% create the image

R = np.zeros((2,2))
R[0,0] = np.cos(theta_camera)
R[0,1] = -np.sin(theta_camera)
R[1,0] = np.sin(theta_camera)
R[1,1] = np.cos(theta_camera)

impts1d = np.linspace(minim,maxim,npts)
impts = np.zeros((npts,2))
impts[:,0] = -focal_length
impts[:,1] = impts1d

impts = np.dot(impts,R.T)

# plot the image

fig,ax = plt.subplots(2,1,figsize=(6,10))

colors = plt.get_cmap('jet')(np.linspace(0, 1.0, npts))
for i in range(npts):
  ax[0].plot(impts[i,0],impts[i,1],'o',color=colors[i])
ax[0].plot(0,0,'s',color='black')
ax[0].axis('equal')

# %% location of points on face 1

# x-coord of first face
xface1 = distface1

# find intersection of the line through impts[i,:] and [0,0] with the line x = x1
miny = np.inf
maxy = -np.inf
face1pts = np.zeros((npts,2))
face1pts[:,0] = xface1
m = impts[:,1]/impts[:,0]
y = impts[:,1]+m*(xface1-impts[:,0])
face1pts[:,1] = y
line1angle = np.arctan2(face1pts[:,1],face1pts[:,0])

# plot points on first face
for i in range(npts):
  ax[0].plot([xface1-1,xface1+1],[face1pts[i,1],face1pts[i,1]],'k:')
  ax[0].plot([impts[i,0],face1pts[i,0]],[impts[i,1],face1pts[i,1]],'-',color=colors[i])
  ax[0].plot(face1pts[i,0],face1pts[i,1],'o',color=colors[i])
ax[0].plot([xface1,xface1],[np.min(face1pts[:,1]),np.max(face1pts[:,1])],'--',color='black')

# %% location of points on face 2 - refraction only

# x-coord of second face, refraction only
xface2 = xface1+distface2

# compute angle after refraction
n = nglass/nair 
# n = n2/n1 = sin(line1angle) / sin(line2angle)
# sin(line2angle) * n = sin(line1angle)
# sin(line2angle) = sin(line1angle)/n
# line2angle = arcsin(sin(line1angle)/n)
line2angle = np.arcsin(np.sin(line1angle)/n)

# equation for line2: 
# y - face1pts[i,1] = tan(line2angle)*(x-face1pts[i,0])
# find intersection with x = x2
face2pts = np.zeros((npts,2))
face2pts[:,0] = xface2
face2pts[:,1] = np.tan(line2angle)*(xface2-face1pts[i,0]) + face1pts[:,1]

# plot
for i in range(npts):
  ax[0].plot([xface2-1,xface2+1],[face2pts[i,1],face2pts[i,1]],'k:')
  ax[0].plot([face1pts[i,0],face2pts[i,0]],[face1pts[i,1],face2pts[i,1]],'-',color=colors[i])
  ax[0].plot(face2pts[i,0],face2pts[i,1],'o',color=colors[i])
ax[0].plot([xface2,xface2],[np.min(face2pts[:,1]),np.max(face2pts[:,1])],'--',color='black')

# %% location of points on face 3 - refraction only

# x-coord of third face, refraction only
xface3 = xface2+distfinal

# compute angle after refraction
line3angle = np.arcsin(np.sin(line2angle)*n)
face3pts = np.zeros((npts,2))
face3pts[:,0] = xface3
face3pts[:,1] = np.tan(line3angle)*(xface3-face2pts[:,0]) + face2pts[:,1]

# plot
for i in range(npts):
  ax[0].plot([face2pts[i,0],face3pts[i,0]],[face2pts[i,1],face3pts[i,1]],'-',color=colors[i])
  ax[0].plot(face3pts[i,0],face3pts[i,1],'o',color=colors[i])
ax[0].plot([xface3,xface3],[np.min(face3pts[:,1]),np.max(face3pts[:,1])],'--',color='black')

# %% plot distances along faces

dorigin1 = np.sqrt((face1pts[:,0]-face1pts[0,0])**2+(face1pts[:,1]-face1pts[0,1])**2) 
dorigin2 = np.sqrt((face2pts[:,0]-face2pts[0,0])**2+(face2pts[:,1]-face2pts[0,1])**2)
dorigin3 = np.sqrt((face3pts[:,0]-face3pts[0,0])**2+(face3pts[:,1]-face3pts[0,1])**2)

ax[1].plot(impts[:,1],dorigin1,'.-',label='face1')
ax[1].plot(impts[:,1],dorigin2,'.-',label='face2')
ax[1].plot(impts[:,1],dorigin3,'o-',label='face3')
ax[1].legend()
ax[1].set_xlabel('image y')
ax[1].set_ylabel('dist')

fig.tight_layout()

# %% prism - plot image and face 1

fig,ax = plt.subplots(2,1,figsize=(6,10))

ax[0].cla()
for i in range(npts):
  ax[0].plot(impts[i,0],impts[i,1],'o',color=colors[i])
ax[0].plot(0,0,'s',color='black')
ax[0].axis('equal')

for i in range(npts):
  ax[0].plot([impts[i,0],face1pts[i,0]],[impts[i,1],face1pts[i,1]],'-',color=colors[i])
  ax[0].plot(face1pts[i,0],face1pts[i,1],'o',color=colors[i])

# %% points along diagonal reflective surface

# prism bottom corner is at y = y1
y1 = face1pts[-1,1]-distprismbottom/np.sqrt(2)

# find intersection between line from face1pts with angle line2angle and the line y - y1 = x - x1
# equation for the line
# y - face1pts[i,1] = tan(line2angle)*(x-face1pts[i,0])
# y - face1pts[i,1] = tan(line2angle)*(y+x1-y1-face1pts[i,0])
# y - tan(line2angle)*y = face1pts[i,1] + tan(line2angle)*(x1-y1-face1pts[i,0])
# y*(1-tan(line2angle)) = face1pts[i,1] + tan(line2angle)*(x1-y1-face1pts[i,0])
face2ptsp = np.zeros((npts,2))
y = (face1pts[:,1] + np.tan(line2angle)*(xface1-y1-face1pts[:,0]))/(1-np.tan(line2angle))
face2ptsp[:,0] = y-y1+xface1
face2ptsp[:,1] = y

# plot
for i in range(npts):
  ax[0].plot([xface1-1,xface1+1],[face1pts[i,1],face1pts[i,1]],'k:')
  ax[0].plot([face1pts[i,0],face2ptsp[i,0]],[face1pts[i,1],face2ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face2ptsp[i,0],face2ptsp[i,1],'o',color=colors[i])

# %% points along top of prism

# top right corner of prism is at x = maxx
maxx = face2ptsp[0,0]+distprismtop/np.sqrt(2)
maxy = y1 + maxx - xface1

# plot the prism sides
ax[0].plot([xface1,xface1],[y1,maxy],'--',color='black')
ax[0].plot([xface1,maxx],[y1,maxy],'--',color='black')
ax[0].plot([xface1,maxx],[maxy,maxy],'--',color='black')

# line3angle is the flip of line2angle over the line x = -y
line3anglep = np.pi/2 - line2angle
face3ptsp = np.zeros((npts,2))
# find intersection of the line that goes from face2ptsp with angle line3anglep with
# the line y = maxy
# y - face2ptsp[:,1] = tan(line3anglep)*(x-face2ptsp[:,0])
# maxy - face2ptsp[:,1] = tan(line3anglep)*(x-face2ptsp[:,0])
# x = (maxy-face2ptsp[:,1])/tan(line3anglep) + face2ptsp[:,0]
face3ptsp[:,0] = (maxy-face2ptsp[:,1])/np.tan(line3anglep) + face2ptsp[:,0]
face3ptsp[:,1] = maxy

# plot
for i in range(npts):
  ax[0].plot([face2ptsp[i,0]-1,face2ptsp[i,0]+1],[face2ptsp[i,1]+1,face2ptsp[i,1]-1],'k:')
  ax[0].plot([face2ptsp[i,0],face3ptsp[i,0]],[face2ptsp[i,1],face3ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face3ptsp[i,0],face3ptsp[i,1],'o',color=colors[i])

# %% plot points along object after the prism

line3anglepperp = np.pi/2 - line3anglep
line4anglepperp = np.arcsin(np.sin(line3anglepperp)*n)
line4anglep = np.pi/2-line4anglepperp
# line that starts at face3ptsp with angle line4anglep and goes to the line y = maxy+y3
# y - face3ptsp[:,1] = tan(line4anglep)*(x-face3ptsp[:,0])
# (y - face3ptsp[:,1])/tan(line4anglep) + face3ptsp[:,0] = x
face4ptsp = np.zeros((npts,2))
face4ptsp[:,0] = (maxy+distfinal-face3ptsp[:,1])/np.tan(line4anglep) + face3ptsp[:,0]
face4ptsp[:,1] = maxy+distfinal

# plot
for i in range(npts):
  ax[0].plot([face3ptsp[i,0],face3ptsp[i,0]],[face3ptsp[i,1]-1,face3ptsp[i,1]+1],'k:')
  ax[0].plot([face3ptsp[i,0],face4ptsp[i,0]],[face3ptsp[i,1],face4ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face4ptsp[i,0],face4ptsp[i,1],'o',color=colors[i])
ax[0].plot([np.min(face4ptsp[:,0]),np.max(face4ptsp[:,0])],[maxy+distfinal,maxy+distfinal],'--',color='black')

# %% plot distances along faces

dorigin1 = np.sqrt((face1pts[:,0]-face1pts[0,0])**2+(face1pts[:,1]-face1pts[0,1])**2) 
dorigin2 = np.sqrt((face2ptsp[:,0]-face2ptsp[0,0])**2+(face2ptsp[:,1]-face2ptsp[0,1])**2)
dorigin3 = np.sqrt((face3ptsp[:,0]-face3ptsp[0,0])**2+(face3ptsp[:,1]-face3ptsp[0,1])**2)
dorigin4 = np.sqrt((face4ptsp[:,0]-face4ptsp[0,0])**2+(face4ptsp[:,1]-face4ptsp[0,1])**2)

ax[1].plot(impts[:,1],dorigin1,'.-',label='face1')
ax[1].plot(impts[:,1],dorigin2,'.-',label='face2')
ax[1].plot(impts[:,1],dorigin3,'.-',label='face3')
ax[1].plot(impts[:,1],dorigin4,'o-',label='face4')
ax[1].legend()
ax[1].set_xlabel('image y')
ax[1].set_ylabel('dist')

fig.tight_layout()

# %% fit projection functions

def fit_projection_matrix(p2d,p1d):
  """ Fit a projection matrix P from 2D points p2d to 1D points p1d
  p2d is n x 2, p1d is n x 1
  Projection matrix P is 3 x 2 and operates on homogeneous coordinates:
  p1dh = p2dh @ P
  Uses the pseudo-inverse to solve the system of equations.
  """
  p2dh = np.c_[p2d,np.ones((p2d.shape[0],1))]
  p1dh = np.c_[p1d,np.ones((p1d.shape[0],1))]
  M = p2dh.T @ p2dh
  cond = np.linalg.cond(M)
  P = np.linalg.inv(M) @ p2dh.T @ p1dh
  return P,cond

def fit_projection_matrix0(p2d,p1d):
  """ Fit a projection matrix P from 2D points p2d to 1D points p1d
  p2d is n x 2, p1d is n x 1
  Projection matrix P is 3 x 2 and operates on homogeneous coordinates:
  p1dh = p2dh @ P
  Uses the pseudo-inverse to solve the system of equations.
  Forces the last row of the projection matrix to be [0,0]."""
  p2dh = p2d
  p1dh = np.c_[p1d,np.ones((p1d.shape[0],1))]
  M = p2dh.T @ p2dh
  cond = np.linalg.cond(M)
  P = np.linalg.inv(M) @ p2dh.T @ p1dh
  P = np.r_[P,np.zeros((1,2))]
  return P,cond

def project(p2d,P):
  """
  Project 2d points p2d to 1d using projection matrix P.
  p2d is n x 2, P is 3 x 2.
  In homogeneous coordinates, p1dh = p2dh @ P
  """
  p2dh = np.c_[p2d,np.ones((p2d.shape[0],1))]
  p1dh = p2dh @ P
  p1d = p1dh[:,0]/p1dh[:,1]
  return p1d

def pretty_print_array(arr):
  print(np.array2string(arr,formatter={'float_kind':lambda x: "%.2f" % x}))

def normalize_points_2d(points):
  """Normalize 2D points to have zero mean and unit standard deviation
  to better condition projection matrix optimzation.
  Inputs 2d points points is n x 2.
  Returns normalized points and the transformation matrix T for homogeneous coordinates.
  """
  mean = np.mean(points, axis=0)
  std = np.std(points)
  T = np.array([[1/std, 0, -mean[0]/std],
                [0, 1/std, -mean[1]/std],
                [0, 0, 1]])
  normalized_points = (points - mean) / std
  return normalized_points, T

def normalize_points_1d(points):
  """Normalize 1D points to have zero mean and unit standard deviation
  to better condition projection matrix optimzation.
  Inputs 1d points points is n x 1.
  Returns normalized points and the transformation matrix T for homogeneous coordinates.
  """
  mean = np.mean(points)
  std = np.std(points)
  T = np.array([[1/std, -mean/std],
                [0, 1]])
  normalized_points = (points - mean) / std
  return normalized_points, T

# from claude
def solve_2d_to_1d_projection(points_2d, points_1d):
  """ Fit a projection matrix P from 2D points points_2d to 1D points points_1d
  points_2d is n x 2, points_1d is n x 1
  Projection matrix P is 3 x 2 and operates on homogeneous coordinates:
  p1dh = p2dh @ P
  Uses lsq to solve the system of equations.
  """
  # Normalize points
  norm_points_2d, T = normalize_points_2d(points_2d)
  norm_points_1d, U = normalize_points_1d(points_1d)
  
  n = len(points_2d)
  A = np.zeros((n, 5))
  b = np.zeros(n)
  
  for i in range(n):
    x, y = norm_points_2d[i]
    u = norm_points_1d[i]
    A[i] = [x, y, 1, -u*x, -u*y]
    b[i] = u
  
  # Solve the system Ap = b
  p = np.linalg.lstsq(A, b, rcond=None)[0]
  
  # Construct the normalized projection matrix
  P_norm = np.array([[p[0], p[1], p[2]],
                      [p[3], p[4], 1]])
  
  # Denormalize the projection matrix
  P = np.linalg.inv(U) @ P_norm @ T
  
  return P.T

from scipy.optimize import least_squares

def project_krt(points_2d, K, R, t):
  """Project 2D points to 1D using the camera parameters."""
  points_proj = np.dot(R, points_2d.T) + t.reshape(-1, 1)
  points_proj = np.dot(K, points_proj)
  points_proj = points_proj[0] / points_proj[1]
  return points_proj

def extract_camera_parameters(params):
  """Extract camera parameters from the optimization result."""
  fx, cx, = params[:2]
  K = np.array([[fx, cx],
                [0, 1]])
  
  c = params[2]
  s = params[3]
  n = np.sqrt(c**2 + s**2)
  c /= n
  s /= n
  t = params[4:6]
  R = np.array([[c,-s],
                [s,c]])
  return K,R,t
  
def pretty_print_camera_parameters(K,R,t):
  print(f'fx: {K[0,0]:.2f}, cx: {K[0,1]:.2f}')
  print(f'theta: {np.arctan2(R[1,0],R[0,0]):.2f}, t: {t[0]:.2f}, {t[1]:.2f}')
  
def KRt_to_projection_matrix(K,R,t):
  """Convert camera parameters to projection matrix."""
  P = K @ np.c_[R,t]
  return P.T

def objective_function(params, points_2d, points_1d):
  """Objective function for optimization."""
  K,R,t = extract_camera_parameters(params)
  projected = project_krt(points_2d, K, R, t)
  residuals = projected - points_1d
  
  return residuals

def calibrate_camera(points_2d, points_1d):
  """ Fit intrinsic camera parameters from corresponding 2D points points_2d and 1D points points_1d
  points_2d is n x 2, points_1d is n x 1
  Projection matrix P is 3 x 2 and operates on homogeneous coordinates
  K,R,t are the intrinsic camera parameters
  p1dh = p2dh @ P
  Uses leastsquares to solve the system of equations, with bounds on the various camera parameters
  """
    
  # Initial guess for camera parameters
  fx = focal_length
  cx = 0
  eps = 1e-6
  params = np.array([fx, cx, 0, 1, 0, 0])
  #lb = np.array([focal_length-eps, cx-eps, -1, -1, -np.inf, -np.inf])
  #ub = np.array([focal_length+eps, cx+eps, 1, 1, np.inf, np.inf])
  lb = np.array([-np.inf, -np.inf, -1, -1, -np.inf, -np.inf])
  ub = np.array([np.inf, np.inf, 1, 1, np.inf, np.inf])

  # Optimize
  result = least_squares(objective_function, params, args=(points_2d, points_1d), bounds=(lb, ub))
  
  # Extract optimized parameters
  K,R,t = extract_camera_parameters(result['x'])
  
  P = KRt_to_projection_matrix(K,R,t)
  
  return P,K,R,t

# %% fit projection matrix

# split samples into train and test  
idxtrain = slice(0,None,2)
idxtest = slice(1,None,2)

# %% what the projection function should be by construction
R1 = np.zeros((3,3))
R1[:-1,:-1] = R
P1true = R1 @ np.array([[0,1],[-focal_length,0],[0,0]])
impts1d_fit_true = project(face1pts,P1true)
errtrue = np.abs(impts1d-impts1d_fit_true)
print('P1true:')
pretty_print_array(P1true)
print(f'Mean error true train: {np.mean(errtrue[idxtrain])}, test: {np.mean(errtrue[idxtest])}')

# %% fit the projection matrix from a volume of points constructed from P1true

# image origin is at impts1d = .5
impt1d_center = np.mean(impts1d)
impt_center = np.zeros((1,2))
impt_center[0,0] = -focal_length
impt_center[0,1] = impt1d_center
impt_center = np.dot(impt_center,R.T)

vol1pts = np.random.rand(100,2)
dx = xface1/2
vol1pts[:,0] = vol1pts[:,0]*(xface1-dx)+dx
vol1pts[:,1] = vol1pts[:,1]*(maxy-y1)+y1

# all imaged points must have a negative dot product with the ray that starts at the origin and goes to the image center
s = np.sum(vol1pts*impt_center,axis=1)
goodidx = s < 0
vol1pts = vol1pts[goodidx,:]
print('Number of points: ',vol1pts.shape[0])

volimpts1d = project(vol1pts,P1true)
P1fit_true,Kfit_true,Rfit_true,tfit_true = calibrate_camera(vol1pts[idxtrain],volimpts1d[idxtrain])
volimpts1d_fit = project(vol1pts,P1fit_true)
err_fit_true = np.abs(volimpts1d-volimpts1d_fit)
print(f'Mean error fit from points constructed with true projection matrix train: {np.mean(err_fit_true[idxtrain])}, test: {np.mean(err_fit_true[idxtest])}')
print('P1fit_true: ')
pretty_print_array(P1fit_true)
pretty_print_camera_parameters(Kfit_true,Rfit_true,tfit_true)

# fig,ax = plt.subplots(1,1,figsize=(6,10))
# volimpts = np.zeros((volimpts1d.shape[0],2))
# volimpts[:,0] = -focal_length
# volimpts[:,1] = volimpts1d
# volimpts = np.dot(volimpts,R.T)
# ax.plot(np.c_[vol1pts[:,0],volimpts[:,0]].T,np.c_[vol1pts[:,1],volimpts[:,1]].T,'-')

# for i in range(npts):
#   ax.plot([xface1-1,xface1+1],[face1pts[i,1],face1pts[i,1]],'k:')
#   ax.plot([impts[i,0],face1pts[i,0]],[impts[i,1],face1pts[i,1]],'-',color=colors[i])
#   ax.plot(face1pts[i,0],face1pts[i,1],'o',color=colors[i])
# ax.plot([xface1,xface1],[np.min(face1pts[:,1]),np.max(face1pts[:,1])],'--',color='black')

# %% try to fit from face1pts - underconstrained -- small error, but not the right matrix
P1fit_face,Kfit_face1,Rfit_face1,tfit_face1 = calibrate_camera(face1pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face = project(face1pts,P1fit_face)
err1_face = np.abs(impts1d-impts1d_fit_face)
print(f'Mean error of matrix fit from face1pts train: {np.mean(err1_face[idxtrain])}, test: {np.mean(err1_face[idxtest])}')
print('P1fit_face: ')
pretty_print_array(P1fit_face)
pretty_print_camera_parameters(Kfit_face1,Rfit_face1,tfit_face1)

# %% sample some image points and construct a volume of possible original points

# find intersection of the line through impts[i,:] and [0,0] with the line x = xsample[i]
nsamples = 100
impts1dsample = np.random.rand(nsamples) # from 0 to 1
imptssample = np.zeros((nsamples,2))
imptssample[:,0] = -focal_length
imptssample[:,1] = impts1dsample
imptssample = np.dot(imptssample,R.T)

xsample = np.random.rand(nsamples)*(xface1-1)+1
sample1pts = np.zeros((nsamples,2))
sample1pts[:,0] = xsample
m = imptssample[:,1]/imptssample[:,0]
y = imptssample[:,1]+m*(xsample-imptssample[:,0])
sample1pts[:,1] = y

P1fit_sample,Kfit_sample,Rfit_sample,tfit_sample = calibrate_camera(sample1pts[idxtrain],impts1dsample[idxtrain])
sampleimpts1d_fit = project(sample1pts,P1fit_sample)
sampleimpts1d_true = project(sample1pts,P1true)
err1_sample = np.abs(impts1dsample-sampleimpts1d_fit)
print(f'Mean error of matrix fit from points samples from before face1 train: {np.mean(err1_sample[idxtrain])}, test: {np.mean(err1_sample[idxtest])}')
print('P1fit_sample: ')
pretty_print_array(P1fit_sample)
pretty_print_camera_parameters(Kfit_sample,Rfit_sample,tfit_sample)

# %% fit from face2pts, refraction only

P2fit_face,Kfit_face2,Rfit_face2,tfit_face2 = calibrate_camera(face2pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face2 = project(face2pts,P2fit_face)
err2_face = np.abs(impts1d-impts1d_fit_face2)
print(f'Mean error of matrix fit from face2pts train: {np.mean(err2_face[idxtrain])}, test: {np.mean(err2_face[idxtest])}')
print('P2fit_face: ')
pretty_print_array(P2fit_face)
pretty_print_camera_parameters(Kfit_face2,Rfit_face2,tfit_face2)

# %% fit from face3pts, refraction only

P3fit_face,Kfit_face3,Rfit_face3,tfit_face3 = calibrate_camera(face3pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face3 = project(face3pts,P3fit_face)
err3_face = np.abs(impts1d-impts1d_fit_face3)
print(f'Mean error of matrix fit from face3pts: {np.mean(err3_face[idxtrain])}, test: {np.mean(err3_face[idxtest])}')
print('P3fit_face: ')
pretty_print_array(P3fit_face)
pretty_print_camera_parameters(Kfit_face3,Rfit_face3,tfit_face3)

# %% fit from face2pts - prism

P2fit_facep,Kfit_face2p,Rfit_face2p,tfit_face2p = calibrate_camera(face2ptsp[idxtrain],impts1d[idxtrain])
impts1d_fit_face2p = project(face2ptsp,P2fit_facep)
err2_facep = np.abs(impts1d-impts1d_fit_face2p)
print(f'Mean error of matrix fit from face2ptsp train: {np.mean(err2_facep[idxtrain])}, test: {np.mean(err2_facep[idxtest])}')
print('P2fit_facep: ')
pretty_print_array(P2fit_facep)
pretty_print_camera_parameters(Kfit_face2p,Rfit_face2p,tfit_face2p)

# %% fit from face3pts - prism

P3fit_facep,Kfit_face3p,Rfit_face3p,tfit_face3p = calibrate_camera(face3ptsp[idxtrain],impts1d[idxtrain])
impts1d_fit_face3p = project(face3ptsp,P3fit_facep)
err3_facep = np.abs(impts1d-impts1d_fit_face3p)
print(f'Mean error of matrix fit from face3ptsp train: {np.mean(err3_facep[idxtrain])}, test: {np.mean(err3_facep[idxtest])}')
print('P3fit_facep: ')
pretty_print_array(P3fit_facep)
pretty_print_camera_parameters(Kfit_face3p,Rfit_face3p,tfit_face3p)

# %% fit from flip(face2pts) - prism

# flip along the x axis
center3p = np.mean(face3ptsp[:,0],axis=0)
face3ptspr = face3ptsp.copy()
face3ptspr[:,0] = 2*center3p-face3ptsp[:,0]

P3fit_facepr,Kfit_face3pr,Rfit_face3pr,tfit_face3pr = calibrate_camera(face3ptspr[idxtrain],impts1d[idxtrain])
impts1d_fit_face3pr = project(face3ptsp,P3fit_facep)
err3_facepr = np.abs(impts1d-impts1d_fit_face3pr)
print(f'Mean error of matrix fit from reflect(face3ptsp) train: {np.mean(err3_facepr[idxtrain])}, test: {np.mean(err3_facepr[idxtest])}')
print('P3fit_facepr: ')
pretty_print_array(P3fit_facepr)
pretty_print_camera_parameters(Kfit_face3pr,Rfit_face3pr,tfit_face3pr)

# %% fit from face4pts - prism

P4fit_facep,Kfit_face4p,Rfit_face4p,tfit_face4p = calibrate_camera(face4ptsp[idxtrain],impts1d[idxtrain])
impts1d_fit_face4p = project(face4ptsp,P4fit_facep)
err4_facep = np.abs(impts1d-impts1d_fit_face4p)
print(f'Mean error of matrix fit from face4ptsp train: {np.mean(err4_facep[idxtrain])}, test: {np.mean(err4_facep[idxtest])}')
print('P4fit_facep: ')
pretty_print_array(P4fit_facep)
pretty_print_camera_parameters(Kfit_face4p,Rfit_face4p,tfit_face4p)

# %% fit from flip(face4pts) - prism

# flip along the x axis
center4p = np.mean(face4ptsp[:,0],axis=0)
face4ptspr = face4ptsp.copy()
face4ptspr[:,0] = 2*center4p-face4ptsp[:,0]

P4fit_facepr,Kfit_face4pr,Rfit_face4pr,tfit_face4pr = calibrate_camera(face4ptspr[idxtrain],impts1d[idxtrain])
impts1d_fit_face4pr = project(face4ptspr,P4fit_facepr)
err4_facepr = np.abs(impts1d-impts1d_fit_face4pr)
print(f'Mean error of matrix fit from reflect(face4ptsp) train: {np.mean(err4_facepr[idxtrain])}, test: {np.mean(err4_facepr[idxtest])}')
print('P4fit_facepr: ')
pretty_print_array(P4fit_facepr)
pretty_print_camera_parameters(Kfit_face4pr,Rfit_face4pr,tfit_face4pr)

with open('/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/data/simulation_coordinates.pkl', 'wb') as f:
  pickle.dump([impts,face3pts,face2pts], f)

plt.show()