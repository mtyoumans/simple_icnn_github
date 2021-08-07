"""Provides a class implementing a prototype
Input Convex Neural Network. 

    Typical usage:
    model = ICNN()
"""


from torch import nn
import torch

class ICNN(nn.Module):
    """ Creates a simple Multi-layer Input Convex Neural Network
    
    Input Convex Neural Networks (ICNNs) are neural networks
    that are convex with respect to the inputs. They are not
    convex with respect to the weights. 
    
    This architecture is based on ideas from (Amos, Xu, Kolter, 2017):

    Amos, Brandon, Lei Xu, and J. Zico Kolter. 
    "Input convex neural networks." 
    International Conference on Machine Learning. 
    PMLR, 2017.
    
    This a prototype implementation of that idea using skip connections
    and constraining weights to be non-negative (or negative in last 
    layer to create a concave network with respect to inputs) using 
    torch.clamp(). As for now, It is to be trained normally with ADAM 
    stochastic gradient descent without any special concern to the weight
    space constraints, though it may be possible to create a better 
    optimizer in the future.
    """
    def __init__(self):
        super(ICNN, self).__init__()
        self.flatten = nn.Flatten()

        self.first_hidden_layer = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU()
        )
        #matrices and nonlinearities for 2nd layer
        self.second_layer_linear_prim = nn.Linear(512,512)
        self.second_layer_linear_prim.weight.data = torch.abs(
            self.second_layer_linear_prim.weight.data)
        self.second_layer_linear_skip = nn.Linear(28*28, 512)
        self.second_layer_act = nn.ReLU()

        #matrices and nonlinearities for 3rd layer
        self.third_layer_linear_prim = nn.Linear(512,512)
        self.third_layer_linear_prim.weight.data = torch.abs(
            self.third_layer_linear_prim.weight.data)
        self.third_layer_linear_skip = nn.Linear(28*28, 512)
        self.third_layer_act = nn.ReLU()

        #matrices and nonlinearities for 4th layer
        self.fourth_layer_linear_prim = nn.Linear(512,512)
        self.fourth_layer_linear_prim.weight.data = torch.abs(
            self.fourth_layer_linear_prim.weight.data)
        self.fourth_layer_linear_skip = nn.Linear(28*28, 512)
        self.fourth_layer_act = nn.ReLU()

        #matrices and nonlinearities for 5th layer
        self.fifth_layer_linear_prim = nn.Linear(512,512)
        self.fifth_layer_linear_prim.weight.data = torch.abs(
            self.fifth_layer_linear_prim.weight.data)
        self.fifth_layer_linear_skip = nn.Linear(28*28, 512)
        self.fifth_layer_act = nn.ReLU()

        #final Output layer
        self.output_layer_linear_prim = nn.Linear(512, 10)
        self.output_layer_linear_prim.weight.data = -1*torch.abs( #check this
            self.output_layer_linear_prim.weight.data)
        self.output_layer_linear_skip = nn.Linear(28*28, 10)
    

    def forward(self, x):
        x = self.flatten(x)
        skip_x2 = x
        skip_x3 = x
        skip_x4 = x
        skip_x5 = x
        skip_x6 = x
        z1 = self.first_hidden_layer(x)
        z1 = self.second_layer_linear_prim(z1)
        z1 = torch.clamp(z1, min = 0, max = None)
        y2 = self.second_layer_linear_skip(skip_x2)
        z2 = self.second_layer_act(z1 + y2)
        z2 = self.third_layer_linear_prim(z2)
        z2 = torch.clamp(z2, min = 0, max = None)
        y3 = self.third_layer_linear_skip(skip_x3)
        z3 = self.third_layer_act(z2 + y3)
        z3 = self.fourth_layer_linear_prim(z3)
        z3 = torch.clamp(z3, min = 0, max = None)
        y4 = self.fourth_layer_linear_skip(skip_x4)
        z4 = self.fourth_layer_act(z3 + y4)
        z4 = self.fifth_layer_linear_prim(z4)
        z4 = torch.clamp(z4, min = 0, max = None)
        y5 = self.fifth_layer_linear_skip(skip_x5)
        z5 = self.fifth_layer_act(z4 + y5)
        z5 = self.output_layer_linear_prim(z5)
        z5 = torch.clamp(z5, min = None, max = 0)#check this
        y6 = self.output_layer_linear_skip(skip_x6)
        logits = z5 + y6
        return logits
