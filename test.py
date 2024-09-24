import torch
import torch.nn as nn
import math

class Plane():
    def __init__(self, alpha=None, beta=None, gamma=None):
        nn.Module.__init__(self)
        if alpha is None:
            alpha = torch.tensor([0.])

        if beta is None:
            beta = torch.tensor([0.])

        if gamma is None:
            gamma = torch.tensor([0.])

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def rotate_plane(self, alpha, beta, gamma):
        rot_mat = torch.tensor([[torch.cos(alpha)*torch.cos(beta), torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma), torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)],
                                [torch.sin(alpha)*torch.cos(beta), torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma), torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)],
                                [-torch.sin(beta), torch.cos(beta)*torch.sin(gamma), torch.cos(beta)*torch.cos(gamma)]])

    def rotate_vec(self, alpha, beta, gamma, vec):
        rot_mat = (
    torch.stack([
        torch.stack([
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma),
            torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
        ]),
        torch.stack([
            torch.sin(alpha) * torch.cos(beta),
            torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma),
            torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
        ]),
        torch.stack([
            -torch.sin(beta),
            torch.cos(beta) * torch.sin(gamma),
            torch.cos(beta) * torch.cos(gamma)
        ])
    ])
)
        return torch.matmul(rot_mat, vec) 
    
    
    
class RefractingPlane(Plane, nn.Module):
    def __init__(self, alpha=None, beta=None, gamma=None, n1=None, n2=None):
        nn.Module.__init__(self)
        Plane.__init__(self, alpha=alpha, beta=beta, gamma=gamma)
        self.n1 = nn.Parameter(torch.tensor(n1), requires_grad=True)
        self.n2 = nn.Parameter(torch.tensor(n2), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)    
        
        

    def forward(self, ray):
        ray.direction = (self.rotate_vec(self.alpha, self.beta, self.gamma, ray.direction)) * self.n1 
        return ray

class Ray():
    def __init__(self, origin=[0,0,0], direction=[1,0,0]):
        """
        Parameters:
        - origin (2-D list): Origin of the ray.
        - direction (2-D list): Direction of the ray.
        """
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=True)
            if len(origin.shape) == 1:
                origin = origin.reshape((3, 1))
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32, requires_grad=True)
            if len(direction.shape) == 1:
                direction = direction.reshape((3, 1))
        
        direction = direction / torch.linalg.vector_norm(direction, dim=0).to(torch.float32)
        self.origin = origin
        self.direction = direction
        self.t = torch.ones((self.direction.shape[1], 1), dtype=torch.float32, device=origin.device)

    def build_ray(self, point1, point2):

        """
        Build a ray from two points.
        Parameters:
        - point1 (2-D list): First point (this will become the origin of the ray).
        - point2 (2-D list): Second point.
        """
        if not isinstance(point1, torch.Tensor):
            point1 = torch.tensor(point1, dtype=torch.float32, requires_grad=True)
            if len(point1.shape) == 1:
                point1 = point1.reshape((3, 1))
        if not isinstance(point2, torch.Tensor):
            point2 = torch.tensor(point2, dtype=torch.float32, requires_grad=True)
            if len(point2.shape) == 1:
                point2 = point2.reshape((3, 1))

        self.origin = point1
        self.direction = (point2 - point1) / torch.linalg.vector_norm(point2 - point1, dim=0)
        self.t = torch.ones((self.direction.shape[1], 1), dtype=torch.float32, device=point1.device)

ray = Ray()
plane1 = RefractingPlane(alpha=0.1, beta=0.2, gamma=0.3, n1=1.0, n2=1.5)
target_ray_direction = torch.tensor([1, 0, 0], dtype=torch.float32, requires_grad=True)[:, None]

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(plane1.parameters(), lr=0.01)
plane1.train()
optimizer.zero_grad()
output_ray = plane1(ray)
loss = criterion(output_ray.direction, target_ray_direction)
loss.backward()