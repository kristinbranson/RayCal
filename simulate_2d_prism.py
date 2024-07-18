# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkAgg')
plt.ion()

# %%
# parameters

# number of points to simulate
npts = 20

# angle of camera
theta_camera = -np.pi/12

# camera focal length
focal_length = 10

# camera image limits
minim = -2
maxim = 2
# image are all at x = -focal_length, y from minim to maxim

nair = 1.0
nglass = 1.5

# distance from the camera to the first face
distface1 = 150

# distance along diagonal from the last ray to the bottom of the prism
distprismbottom = 5
# distance along diagonal from the first ray to the top of the prism
distprismtop = 2

# for refractive only case, thickness of the glass
distface2 = 20

# distance from the planar object to the last face
# making this fly-height is too small compared to the prism currently
distfinal = 20

# constraints on camera parameters
eps = 1e-6
fxmin = focal_length - eps
fxmax = focal_length + eps
cxmin = -eps
cxmax = eps

plotnormallength = distface2/5

# %%
# create the image

Rcamera = np.zeros((2,2))
Rcamera[0,0] = np.cos(theta_camera)
Rcamera[0,1] = -np.sin(theta_camera)
Rcamera[1,0] = np.sin(theta_camera)
Rcamera[1,1] = np.cos(theta_camera)

impts1d = np.linspace(minim,maxim,npts)

def camera_1d_to_2d(impts1d):
  impts = np.zeros((len(impts1d),2))
  impts[:,0] = -focal_length
  impts[:,1] = impts1d

  impts = np.dot(impts,Rcamera.T)
  return impts

impts = camera_1d_to_2d(impts1d)

# plot the image

fig,ax = plt.subplots(2,1,figsize=(6,10))

colors = plt.get_cmap('jet')(np.linspace(0, 1.0, npts))
for i in range(npts):
  ax[0].plot(impts[i,0],impts[i,1],'o',color=colors[i])
ax[0].plot(0,0,'s',color='black')
_ = ax[0].axis('equal')


# %%
# cribbed from claude

def intersection_lines(line1start, line1end=None, line1angle=None, issegment1=None,
                       line2start=None, line2end=None, line2angle=None, issegment2=None):
  
  n_lines = max(len(line1start) if line1start is not None else 1,
                len(line2start) if line2start is not None else 1,
                len(line1end) if line1end is not None else 1,
                len(line2end) if line2end is not None else 1,
                len(line1angle) if line1angle is not None else 1,
                len(line2angle) if line2angle is not None else 1)
  
  def validate_points(points, name):
    try:
      arr = np.array(points, dtype=float)
      if arr.ndim == 1 and len(arr) == 2:
        arr = arr.reshape(1, 2)
      elif arr.ndim == 2 and arr.shape[1] == 2:
        pass
      else:
          raise ValueError(f"{name} must be a 2D point or array of 2D points")
      if arr.shape[0] == 1:
        arr = np.broadcast_to(arr, (n_lines, 2))
      return arr
    except:
      raise ValueError(f"{name} must be a tuple, list, or numpy array with shape (2,) or (n,2)")
    
  def validate_angle(angle, name):
    try:
      arr = np.array(angle, dtype=float)
      if arr.ndim == 0:
        arr = arr.reshape(1)
      elif arr.ndim > 1:
        arr = arr.flatten()
      if arr.shape[0] == 1:
        arr = np.broadcast_to(arr, (n_lines,))
      return arr
    except:
      raise ValueError(f"{name} must be a scalar, list, or numpy array with shape (1,) or (n,)")

  # Validate and convert inputs
  line1start = validate_points(line1start, "line1start")
  if line2start is not None:
      line2start = validate_points(line2start, "line2start")
  
  if line1end is not None:
      line1end = validate_points(line1end, "line1end")
  elif line1angle is not None:
      line1angle = validate_angle(line1angle, "line1angle")
  else:
      raise ValueError("Must provide either line1end or line1angle")
  
  if line2end is not None:
      line2end = validate_points(line2end, "line2end")
  elif line2angle is not None:
      line2angle = validate_angle(line2angle, "line2angle")
  else:
      raise ValueError("Must provide either line2end or line2angle")

  # Calculate direction vectors
  if line1end is not None:
      line1end = np.broadcast_to(line1end, (n_lines, 2))
      dir1 = line1end - line1start
  else:
      dir1 = np.column_stack([np.cos(line1angle), np.sin(line1angle)])

  if line2end is not None:
      line2end = np.broadcast_to(line2end, (n_lines, 2))
      dir2 = line2end - line2start
  else:
      dir2 = np.column_stack([np.cos(line2angle), np.sin(line2angle)])

  # Calculate the denominator of the intersection formula
  det = np.cross(dir1, dir2)

  # Check if lines are parallel
  parallel = np.abs(det) < 1e-8
  
  # Calculate the intersection points
  t = np.cross(line2start - line1start, dir2) / det
  intersections = line1start + t[:,None] * dir1

  # Check if intersections are within segments (if applicable)
  if issegment1:
      t1 = np.sum((intersections - line1start) * dir1, axis=1) / np.sum(dir1 * dir1, axis=1)
      valid1 = (t1 >= 0) & (t1 <= 1)
  else:
      valid1 = np.ones(n_lines, dtype=bool)

  if issegment2:
      t2 = np.sum((intersections - line2start) * dir2, axis=1) / np.sum(dir2 * dir2, axis=1)
      valid2 = (t2 >= 0) & (t2 <= 1)
  else:
      valid2 = np.ones(n_lines, dtype=bool)

  # Combine all validity checks
  valid = ~parallel & valid1 & valid2

  # Set invalid intersections to NaN
  intersections[~valid] = np.nan

  return intersections

# Example usage:
# Single intersection
point1 = intersection_lines([(0, 0)], line1end=[(1, 1)], line2start=[(0, 1)], line2end=[(1, 0)])
print("Single Intersection:", point1)

# Multiple intersections
lines1_start = np.array([(0, 0), (1, 1), (2, 2)])
lines1_end = np.array([(1, 1), (2, 2), (3, 3)])
lines2_start = np.array([(0, 1), (1, 2), (2, 3)])
lines2_end = np.array([(1, 0), (2, 1), (3, 2)])

points = intersection_lines(lines1_start, line1end=lines1_end, 
                            line2start=lines2_start, line2end=lines2_end)
print("Multiple Intersections:")
print(points)

# Using angles
angles1 = np.array([np.pi/4, np.pi/3, np.pi/6])
angles2 = np.array([-np.pi/4, -np.pi/3, -np.pi/6])

points_angles = intersection_lines(lines1_start, line1angle=angles1,
                                   line2start=lines2_start, line2angle=angles2)
print("Intersections with angles:")
print(points_angles)


# %%
# location of points on face 1

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

face1pts = intersection_lines(line1start=[0,0],
                              line1end=impts,
                              issegment1=False,
                              line2start=[xface1,0],
                              line2end=[xface1,1],
                              issegment2=False)

# plot points on first face
for i in range(npts):
  ax[0].plot([xface1-plotnormallength,xface1+plotnormallength],[face1pts[i,1],face1pts[i,1]],'k:')
  ax[0].plot([impts[i,0],face1pts[i,0]],[impts[i,1],face1pts[i,1]],'-',color=colors[i])
  ax[0].plot(face1pts[i,0],face1pts[i,1],'o',color=colors[i])
_ = ax[0].plot([xface1,xface1],[np.min(face1pts[:,1]),np.max(face1pts[:,1])],'--',color='black')

# %%
# location of points on face 2 - refraction only

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

face2pts = intersection_lines(line1start=face1pts,
                              line1angle=line2angle,
                              issegment1=False,
                              line2start=[xface2,0],
                              line2end=[xface2,1],
                              issegment2=False)

# plot
for i in range(npts):
  ax[0].plot([xface2-plotnormallength,xface2+plotnormallength],[face2pts[i,1],face2pts[i,1]],'k:')
  ax[0].plot([face1pts[i,0],face2pts[i,0]],[face1pts[i,1],face2pts[i,1]],'-',color=colors[i])
  ax[0].plot(face2pts[i,0],face2pts[i,1],'.',color=colors[i])
_ = ax[0].plot([xface2,xface2],[np.min(face2pts[:,1]),np.max(face2pts[:,1])],'--',color='black')

# %%
# location of points on face 3 - refraction only

# x-coord of third face, refraction only
xface3 = xface2+distfinal

# compute angle after refraction
line3angle = np.arcsin(np.sin(line2angle)*n)

face3pts = intersection_lines(line1start=face2pts,
                              line1angle=line3angle,
                              issegment1=False,
                              line2start=[xface3,0],
                              line2end=[xface3,1],
                              issegment2=False)

# plot
for i in range(npts):
  ax[0].plot([face2pts[i,0],face3pts[i,0]],[face2pts[i,1],face3pts[i,1]],'-',color=colors[i])
  ax[0].plot(face3pts[i,0],face3pts[i,1],'.',color=colors[i])
_ = ax[0].plot([xface3,xface3],[np.min(face3pts[:,1]),np.max(face3pts[:,1])],'--',color='black')

# %%
# plot distances along faces

dorigin1 = np.sqrt((face1pts[:,0]-face1pts[0,0])**2+(face1pts[:,1]-face1pts[0,1])**2) 
dorigin2 = np.sqrt((face2pts[:,0]-face2pts[0,0])**2+(face2pts[:,1]-face2pts[0,1])**2)
dorigin3 = np.sqrt((face3pts[:,0]-face3pts[0,0])**2+(face3pts[:,1]-face3pts[0,1])**2)

ax[1].plot(impts[:,1],dorigin1,'.-',label='Air->glass interface')
ax[1].plot(impts[:,1],dorigin2,'.-',label='Glass->air interface')
ax[1].plot(impts[:,1],dorigin3,'o-',label='After refraction')
ax[1].legend()
ax[1].set_xlabel('image y')
ax[1].set_ylabel('dist')

fig.tight_layout()

# %%
# prism - plot image and face 1

fig,ax = plt.subplots(2,1,figsize=(6,10))

ax[0].cla()
for i in range(npts):
  ax[0].plot(impts[i,0],impts[i,1],'o',color=colors[i])
ax[0].plot(0,0,'s',color='black')
ax[0].axis('equal')

for i in range(npts):
  ax[0].plot([impts[i,0],face1pts[i,0]],[impts[i,1],face1pts[i,1]],'-',color=colors[i])
  ax[0].plot(face1pts[i,0],face1pts[i,1],'o',color=colors[i])

# %%
# points along diagonal reflective surface

# prism bottom corner is at y = y1
y1 = face1pts[-1,1]-distprismbottom/np.sqrt(2)

# find intersection between line from face1pts with angle line2angle and the line y - y1 = x - xface1
# equation for the line
face2ptsp = intersection_lines(line1start=face1pts,
                              line1angle=line2angle,
                              issegment1=False,
                              line2start=[xface1,y1],
                              line2angle=[np.pi/4,],
                              issegment2=False)

# plot
for i in range(npts):
  ax[0].plot([xface1-plotnormallength,xface1+plotnormallength],[face1pts[i,1],face1pts[i,1]],'k:')
  ax[0].plot([face1pts[i,0],face2ptsp[i,0]],[face1pts[i,1],face2ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face2ptsp[i,0],face2ptsp[i,1],'o',color=colors[i])

# %%
# points along top of prism

# top right corner of prism is at x = maxx
maxx = face2ptsp[0,0]+distprismtop/np.sqrt(2)
maxy = y1 + maxx - xface1

# plot the prism sides
ax[0].plot([xface1,xface1],[y1,maxy],'--',color='black')
ax[0].plot([xface1,maxx],[y1,maxy],'--',color='black')
ax[0].plot([xface1,maxx],[maxy,maxy],'--',color='black')

# line3angle is the flip of line2angle over the line x = -y
line3anglep = np.pi/2 - line2angle

face3ptsp = intersection_lines(line1start=face2ptsp,
                              line1angle=line3anglep,
                              issegment1=False,
                              line2start=[xface1,maxy],
                              line2end=[maxx,maxy],
                              issegment2=True)

# plot
for i in range(npts):
  ax[0].plot([face2ptsp[i,0]-plotnormallength,face2ptsp[i,0]+plotnormallength],[face2ptsp[i,1]+1,face2ptsp[i,1]-1],'k:')
  ax[0].plot([face2ptsp[i,0],face3ptsp[i,0]],[face2ptsp[i,1],face3ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face3ptsp[i,0],face3ptsp[i,1],'.',color=colors[i])

# %%
# plot points along object after the prism

line3anglepperp = np.pi/2 - line3anglep
line4anglepperp = np.arcsin(np.sin(line3anglepperp)*n)
line4anglep = np.pi/2-line4anglepperp
# line that starts at face3ptsp with angle line4anglep and goes to the line y = maxy+y3

# y - face3ptsp[:,1] = tan(line4anglep)*(x-face3ptsp[:,0])
# (y - face3ptsp[:,1])/tan(line4anglep) + face3ptsp[:,0] = x
face4ptsp = np.zeros((npts,2))
face4ptsp[:,0] = (maxy+distfinal-face3ptsp[:,1])/np.tan(line4anglep) + face3ptsp[:,0]
face4ptsp[:,1] = maxy+distfinal

face4ptsp = intersection_lines(line1start=face3ptsp,
                              line1angle=line4anglep,
                              issegment1=False,
                              line2start=[0,maxy+distfinal],
                              line2end=[1,maxy+distfinal],
                              issegment2=False)
# plot
for i in range(npts):
  ax[0].plot([face3ptsp[i,0],face3ptsp[i,0]],[face3ptsp[i,1]-plotnormallength,face3ptsp[i,1]+plotnormallength],'k:')
  ax[0].plot([face3ptsp[i,0],face4ptsp[i,0]],[face3ptsp[i,1],face4ptsp[i,1]],'-',color=colors[i])
  ax[0].plot(face4ptsp[i,0],face4ptsp[i,1],'.',color=colors[i])
_ = ax[0].plot([np.min(face4ptsp[:,0]),np.max(face4ptsp[:,0])],[maxy+distfinal,maxy+distfinal],'--',color='black')

# %%
# plot distances along faces

dorigin1 = np.sqrt((face1pts[:,0]-face1pts[0,0])**2+(face1pts[:,1]-face1pts[0,1])**2) 
dorigin2 = np.sqrt((face2ptsp[:,0]-face2ptsp[0,0])**2+(face2ptsp[:,1]-face2ptsp[0,1])**2)
dorigin3 = np.sqrt((face3ptsp[:,0]-face3ptsp[0,0])**2+(face3ptsp[:,1]-face3ptsp[0,1])**2)
dorigin4 = np.sqrt((face4ptsp[:,0]-face4ptsp[0,0])**2+(face4ptsp[:,1]-face4ptsp[0,1])**2)

ax[1].plot(impts[:,1],dorigin1,'.-',label='Air->glass interface')
ax[1].plot(impts[:,1],dorigin2,'.-',label='Mirror')
ax[1].plot(impts[:,1],dorigin3,'.-',label='Glass->air interface')
ax[1].plot(impts[:,1],dorigin4,'o-',label='After prism')
ax[1].legend()
ax[1].set_xlabel('image y')
ax[1].set_ylabel('dist')

fig.tight_layout()

# %%
# fit projection functions

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
  params = np.array([fx, cx, 0, 1, 0, 0])
  #lb = np.array([focal_length-eps, cx-eps, -1, -1, -np.inf, -np.inf])
  #ub = np.array([focal_length+eps, cx+eps, 1, 1, np.inf, np.inf])
  lb = np.array([fxmin, cxmin, -1, -1, -np.inf, -np.inf])
  ub = np.array([fxmax, cxmax, 1, 1, np.inf, np.inf])

  # Optimize
  result = least_squares(objective_function, params, args=(points_2d, points_1d), bounds=(lb, ub))
  
  # Extract optimized parameters
  K,R,t = extract_camera_parameters(result['x'])
  
  P = KRt_to_projection_matrix(K,R,t)
  
  return P,K,R,t

def pretty_print_proj_info(projinfo):
  print(f'{projinfo["name"]}:')
  if 'P' in projinfo:
    print('Proj matrix:')
    pretty_print_array(projinfo['P'])
  if 'K' in projinfo:
    pretty_print_camera_parameters(projinfo['K'],projinfo['R'],projinfo['t'])
  if 'err_train' in projinfo:
    print(f'Mean error train: {projinfo["err_train"]}')
  if 'err_test' in projinfo:
    print(f'Mean error test: {projinfo["err_test"]}')
  print('')

# %%
# split samples into train and test  
idxtrain = slice(0,None,2)
idxtest = slice(1,None,2)

projinfo = {}

# %%
# what the projection function should be by construction

projinfo['true'] = {'name': 'True projection, face1'}

R1 = np.zeros((3,3))
R1[:-1,:-1] = Rcamera
Ptrue_face1 = R1 @ np.array([[0,1],[-focal_length,0],[0,0]])
projinfo['true']['P'] = Ptrue_face1

impts1d_fit_true = project(face1pts,Ptrue_face1)
projinfo['true']['err_train'] = np.mean(np.abs(impts1d[idxtrain]-impts1d_fit_true[idxtrain]))
projinfo['true']['err_test'] = np.mean(np.abs(impts1d[idxtest]-impts1d_fit_true[idxtest]))
pretty_print_proj_info(projinfo['true'])

# %%
# fit the projection matrix from a volume of points constructed from P1true

# image origin is at impts1d = .5
impt1d_center = np.mean(impts1d)
impt_center = np.zeros((1,2))
impt_center[0,0] = -focal_length
impt_center[0,1] = impt1d_center
impt_center = np.dot(impt_center,Rcamera.T)

pre_face1_pts_ptrue = np.random.rand(100,2)
dx = xface1/2
pre_face1_pts_ptrue[:,0] = pre_face1_pts_ptrue[:,0]*(xface1-dx)+dx
pre_face1_pts_ptrue[:,1] = pre_face1_pts_ptrue[:,1]*(maxy-y1)+y1

# all imaged points must have a negative dot product with the ray that starts at the origin and goes to the image center
s = np.sum(pre_face1_pts_ptrue*impt_center,axis=1)
goodidx = s < 0
pre_face1_pts_ptrue = pre_face1_pts_ptrue[goodidx,:]
print('Number of points: ',pre_face1_pts_ptrue.shape[0])

projinfo['pre_face1_ptrue'] = {'name': 'Correspondences from pre-face1 points, created using Ptrue_face1'}

impts1d_pre_face1_ptrue = project(pre_face1_pts_ptrue,Ptrue_face1)
projinfo['pre_face1_ptrue']['P'],projinfo['pre_face1_ptrue']['K'],projinfo['pre_face1_ptrue']['R'],projinfo['pre_face1_ptrue']['t'] = \
  calibrate_camera(pre_face1_pts_ptrue[idxtrain],impts1d_pre_face1_ptrue[idxtrain])
impts1d_pre_face1_ptrue_fit = project(pre_face1_pts_ptrue,projinfo['pre_face1_ptrue']['P'])
projinfo['pre_face1_ptrue']['err_train'] = np.mean(np.abs(impts1d_pre_face1_ptrue[idxtrain]-impts1d_pre_face1_ptrue[idxtrain]))
projinfo['pre_face1_ptrue']['err_test'] = np.mean(np.abs(impts1d_pre_face1_ptrue[idxtest]-impts1d_pre_face1_ptrue[idxtest]))
pretty_print_proj_info(projinfo['pre_face1_ptrue'])


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

# %%
# try to fit from face1pts

projinfo['face1'] = {'name': 'Face1'}
projinfo['face1']['P'],projinfo['face1']['K'],projinfo['face1']['R'],projinfo['face1']['t'] = \
  calibrate_camera(face1pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face1 = project(face1pts,projinfo['face1']['P'])
projinfo['face1']['err_train'] = np.mean(np.abs(impts1d[idxtrain]-impts1d_fit_face1[idxtrain]))
projinfo['face1']['err_test'] = np.mean(np.abs(impts1d[idxtest]-impts1d_fit_face1[idxtest]))
pretty_print_proj_info(projinfo['face1'])

# %%
# sample more image points and construct a volume of possible original points

# find intersection of the line through impts[i,:] and [0,0] with the line x = xsample[i]
nsamples = 100
impts1d_pre_face1_geom = np.random.rand(nsamples)*(maxim-minim)+minim
impts_more = camera_1d_to_2d(impts1d_pre_face1_geom)

# where along the rays to sample
xsample = np.random.rand(nsamples)*(xface1-1)+1
pre_face1_pts_geom = intersection_lines(line1start=[0,0],
                                        line1end=impts_more,
                                        issegment1=False,
                                        line2start=np.c_[xsample,np.zeros(nsamples)],
                                        line2end=np.c_[xsample,np.ones(nsamples)],
                                        issegment2=False)

projinfo['pre_face1_geom'] = {'name': 'Correspondences from pre-face1 points, created using geometry'}

projinfo['pre_face1_geom']['P'],projinfo['pre_face1_geom']['K'],projinfo['pre_face1_geom']['R'],projinfo['pre_face1_geom']['t'] = \
  calibrate_camera(pre_face1_pts_geom[idxtrain],impts1d_pre_face1_geom[idxtrain])
impts1d_pre_face1_geom = project(pre_face1_pts_geom,projinfo['pre_face1_geom']['P'])
projinfo['pre_face1_geom']['err_train'] = np.mean(np.abs(impts1d_pre_face1_geom[idxtrain]-impts1d_pre_face1_geom[idxtrain]))
projinfo['pre_face1_geom']['err_test'] = np.mean(np.abs(impts1d_pre_face1_geom[idxtest]-impts1d_pre_face1_geom[idxtest]))
pretty_print_proj_info(projinfo['pre_face1_geom'])

# %%
# fit from face2pts, refraction only

projinfo['face2pts_refraction'] = {'name': 'Face2 - refraction only'}
projinfo['face2pts_refraction']['P'],projinfo['face2pts_refraction']['K'],projinfo['face2pts_refraction']['R'],projinfo['face2pts_refraction']['t'] = \
  calibrate_camera(face2pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face2 = project(face2pts,projinfo['face2pts_refraction']['P'])
projinfo['face2pts_refraction']['err_train'] = np.mean(np.abs(impts1d[idxtrain]-impts1d_fit_face2[idxtrain]))
projinfo['face2pts_refraction']['err_test'] = np.mean(np.abs(impts1d[idxtest]-impts1d_fit_face2[idxtest]))
pretty_print_proj_info(projinfo['face2pts_refraction'])

# %%
# fit from face3pts, refraction only

projinfo['face3pts_refraction'] = {'name': 'Face3 - refraction only'}
projinfo['face3pts_refraction']['P'],projinfo['face3pts_refraction']['K'],projinfo['face3pts_refraction']['R'],projinfo['face3pts_refraction']['t'] = \
  calibrate_camera(face3pts[idxtrain],impts1d[idxtrain])
impts1d_fit_face3 = project(face3pts,projinfo['face3pts_refraction']['P'])
projinfo['face3pts_refraction']['err_train'] = np.mean(np.abs(impts1d[idxtrain]-impts1d_fit_face3[idxtrain]))
projinfo['face3pts_refraction']['err_test'] = np.mean(np.abs(impts1d[idxtest]-impts1d_fit_face3[idxtest]))
pretty_print_proj_info(projinfo['face3pts_refraction'])

# %%
# fit from face2pts - prism

projinfo['face2pts_prism'] = {'name': 'Prism diagonal face'}
projinfo['face2pts_prism']['P'],projinfo['face2pts_prism']['K'],projinfo['face2pts_prism']['R'],projinfo['face2pts_prism']['t'] = \
  calibrate_camera(face2ptsp[idxtrain],impts1d[idxtrain])
impts1d_fit_face2_prism = project(face2ptsp,projinfo['face2pts_prism']['P'])
projinfo['face2pts_prism']['err_train'] = np.mean(np.abs(impts1d[idxtrain]-impts1d_fit_face2_prism[idxtrain]))
projinfo['face2pts_prism']['err_test'] = np.mean(np.abs(impts1d[idxtest]-impts1d_fit_face2_prism[idxtest]))
pretty_print_proj_info(projinfo['face2pts_prism'])

# %%
# fit from face3pts - prism

isvalid_face3 = ~np.isnan(face3ptsp).any(axis=1)
face3ptsp_valid = face3ptsp[isvalid_face3]
impts1d_valid_face3 = impts1d[isvalid_face3]

projinfo['face3pts_prism'] = {'name': 'Prism top face'}
projinfo['face3pts_prism']['P'],projinfo['face3pts_prism']['K'],projinfo['face3pts_prism']['R'],projinfo['face3pts_prism']['t'] = \
  calibrate_camera(face3ptsp_valid[idxtrain],impts1d_valid_face3[idxtrain])
impts1d_fit_face3_prism_valid = project(face3ptsp_valid,projinfo['face3pts_prism']['P'])
projinfo['face3pts_prism']['err_train'] = np.mean(np.abs(impts1d_valid_face3[idxtrain]-impts1d_fit_face3_prism_valid[idxtrain]))
projinfo['face3pts_prism']['err_test'] = np.mean(np.abs(impts1d_valid_face3[idxtest]-impts1d_fit_face3_prism_valid[idxtest]))
pretty_print_proj_info(projinfo['face3pts_prism'])

# %%
# fit to face3ptsp and flip(impts1d) - prism

# flip 
impts1d_valid_face3_flip = -impts1d_valid_face3
projinfo['face3pts_prism_flip'] = {'name': 'Prism top face, image flipped'}
projinfo['face3pts_prism_flip']['P'],projinfo['face3pts_prism_flip']['K'],projinfo['face3pts_prism_flip']['R'],projinfo['face3pts_prism_flip']['t'] = \
  calibrate_camera(face3ptsp_valid[idxtrain],impts1d_valid_face3_flip[idxtrain])
impts1d_fit_face3_prism_flip_valid = project(face3ptsp_valid,projinfo['face3pts_prism_flip']['P'])
projinfo['face3pts_prism_flip']['err_train'] = np.mean(np.abs(impts1d_valid_face3_flip[idxtrain]-impts1d_fit_face3_prism_flip_valid[idxtrain]))
projinfo['face3pts_prism_flip']['err_test'] = np.mean(np.abs(impts1d_valid_face3_flip[idxtest]-impts1d_fit_face3_prism_flip_valid[idxtest]))
pretty_print_proj_info(projinfo['face3pts_prism_flip'])

# %%
# fit from face4pts - prism

isvalid_face4 = ~np.isnan(face4ptsp).any(axis=1)
face4ptsp_valid = face4ptsp[isvalid_face4]
impts1d_valid_face4 = impts1d[isvalid_face4]

projinfo['face4pts_prism'] = {'name': 'Above prism'}
projinfo['face4pts_prism']['P'],projinfo['face4pts_prism']['K'],projinfo['face4pts_prism']['R'],projinfo['face4pts_prism']['t'] = \
  calibrate_camera(face4ptsp_valid[idxtrain],impts1d_valid_face4[idxtrain])
impts1d_fit_face4_prism_valid = project(face4ptsp_valid,projinfo['face4pts_prism']['P'])
projinfo['face4pts_prism']['err_train'] = np.mean(np.abs(impts1d_valid_face4[idxtrain]-impts1d_fit_face4_prism_valid[idxtrain]))
projinfo['face4pts_prism']['err_test'] = np.mean(np.abs(impts1d_valid_face4[idxtest]-impts1d_fit_face4_prism_valid[idxtest]))
pretty_print_proj_info(projinfo['face4pts_prism'])

# %%
# fit from impts_flip to face4pts - prism

impts1d_valid_face4_flip = -impts1d_valid_face4
projinfo['face4pts_prism_flip'] = {'name': 'Above prism, image flipped'}
projinfo['face4pts_prism_flip']['P'],projinfo['face4pts_prism_flip']['K'],projinfo['face4pts_prism_flip']['R'],projinfo['face4pts_prism_flip']['t'] = \
  calibrate_camera(face4ptsp_valid[idxtrain],impts1d_valid_face4_flip[idxtrain])
impts1d_fit_face4_prism_flip_valid = project(face4ptsp_valid,projinfo['face4pts_prism_flip']['P'])
projinfo['face4pts_prism_flip']['err_train'] = np.mean(np.abs(impts1d_valid_face4_flip[idxtrain]-impts1d_fit_face4_prism_flip_valid[idxtrain]))
projinfo['face4pts_prism_flip']['err_test'] = np.mean(np.abs(impts1d_valid_face4_flip[idxtest]-impts1d_fit_face4_prism_flip_valid[idxtest]))
pretty_print_proj_info(projinfo['face4pts_prism_flip'])

# %% 
# compare

from tabulate import tabulate

print(tabulate([ [v['name'], v['err_train'], v['err_test']] for v in projinfo.values() ],
               headers=['', 'Train error', 'Test error']))

table = []
for v in projinfo.values():
  row = [v['name'],]
  if 'K' in v:
    row.extend([v['K'][0,0], v['K'][0,1], np.arctan2(v['R'][1,0],v['R'][0,0]), v['t'][0], v['t'][1]])
    table.append(row)

print(tabulate(table, headers=['', 'fx', 'cx', 'theta', 'tx', 'ty']))

# %% 
# plot camera locations

# fig,ax = plt.subplots(1,1,figsize=(6,10))
# ax.plot(0,0,'ks-',label='True')
# ax.plot([0,np.cos(theta_camera)],[0,np.sin(theta_camera)],'k-')
# ax.plot()
# for v in projinfo.values():
#   if 'K' in v:
#     theta = np.arctan2(v['R'][1,0],v['R'][0,0])
#     t = v['t']
#     ax.plot(t[0],t[1],'o-',label=v['name'])
#     ax.plot([t[0],t[0]+np.cos(theta)],[t[1],t[1]+np.sin(theta)],'-')

# ax.legend()

# %%
plt.show()
