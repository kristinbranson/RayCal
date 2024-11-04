import torch

# Euclidean distance loss function
def euclidean_distance(output, label):
    """
    Arrange data points along dim=0 and components along dim=1
    """
    distance = torch.norm(output - label, 
                        p=2, 
                        dim=0)
    return distance