import torch
import torch.nn as nn

# Define Class1 with an nn.Parameter
class Class1(nn.Module):
    def __init__(self):
        super(Class1, self).__init__()
        # Initialize the parameter w
        self.w = nn.Parameter(torch.randn(10, 10))  # Example parameter
        # Create an instance of Class2, passing w to it
        self.class2_instance = Class2(self.w)
        print(self.class2_instance.w1)

    def forward(self, x):
        # Use the class2_instance in the forward pass
        return self.class2_instance(x)

class Parent(nn.Module):
    def __init__(self, w1):
        super(Parent, self).__init__()
        self.w1 = w1
    

# Define Class2 that will take a parameter
class Class2(Parent, nn.Module):
    def __init__(self, w1):
        super(Class2, self).__init__(w1=w1)
        # Store the passed parameter as an attribute
        
    def forward(self, x):
        # Use the parameter in some operation
        return torch.mm(x, self.w1)



# Example usage
class1_instance = Class1()  # Create an instance of Class1

# Now you can use class1_instance
input_tensor = torch.randn(5, 10)  # Example input
output = class1_instance(input_tensor)  # Forward pass
