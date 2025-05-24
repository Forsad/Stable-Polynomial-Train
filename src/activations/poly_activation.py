import torch
import torch.nn as nn
from functools import partial
from functools import lru_cache
from torch.nn.functional import conv2d
import numpy as np
from scipy import linalg
class EpochAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._epoch = 0
        
    @property
    def epoch(self):
        return self._epoch
        
    @epoch.setter
    def epoch(self, value):
        self._epoch = value

#Create a base class for the activation function with boundary loss
#The class should have a method to calculate the boundary loss
class PolyActivation(EpochAwareModule):
    def __init__(self, B, boundary_loss_params):
        # print(boundary_loss_params)
        super().__init__()
        self.b_type = boundary_loss_params['type']
        if self.b_type in ['exp', 'l2']:
            self.penalty_B = boundary_loss_params['penalty_B']
        elif self.b_type not in ['maxnorm']:
            raise Exception("Unknow boundary loss type " + self.b_type)
        

        self.boundary_loss = 0.0
        

    def get_exp_boundary_loss(self, x):
        # Calculate penalty for values exceeding B
        excess = torch.relu(torch.abs(x) - self.penalty_B)
        return torch.mean(torch.exp(excess) - 1)
    
    def get_l2_boundary_loss(self, x):
        excess = torch.relu(torch.abs(x) - self.penalty_B)
        return torch.mean(excess**2)

    def get_max_boundary_loss(self, x):
        return torch.max(torch.abs(x))

    def get_boundary_loss(self, x):
        if self.b_type == 'exp':
            return self.get_exp_boundary_loss(x)
        elif self.b_type == 'l2':
            return self.get_l2_boundary_loss(x)
        return self.get_max_boundary_loss(x)
    
    
class PolyFit(PolyActivation):
    def __init__(self, activation:str, B, samp_size, pol_degree, learnable_coeff,
        initialization, boundary_loss_params):
        print("B", B, "samp_size", samp_size, "initialization", initialization, "pol_degree", pol_degree)
        super(PolyFit, self).__init__(B, boundary_loss_params)
        coefficients = None
        if initialization == 'least_square':
            coefficients = self.least_square_fit(activation, B, samp_size, pol_degree)
        elif initialization == 'remez':
            coefficients = self.remez_fit(activation, B, samp_size, pol_degree)
        else:
            raise ValueError(f'Unknown initialization: {initialization}')

        self.register_buffer('coefficients', torch.from_numpy(coefficients).float().requires_grad_(learnable_coeff))
        self.register_buffer('exponents', torch.arange(len(coefficients) - 1, -1, -1).requires_grad_(False))
        self.current_boundary_loss = 0.0
    def get_activation_points(self, activation, x_points):
        y_points = None
        if activation == 'silu':
            y_points = [x / (1 + np.exp(-x)) for x in x_points]
        elif activation == 'relu':
            y_points = [max(0, x) for x in x_points]
        else:
            raise ValueError(f'Unknown activation: {activation}')
        return y_points

    def remez_fit(self, activation, B, samp_size, pol_degree):
        # 1. Choose initial set of control points (n+2 points for degree n)
        x_points = np.linspace(-B, B, samp_size)
        y_points = self.get_activation_points(activation, x_points)

        n = len(x_points)
        if pol_degree == -1:
            pol_degree = n - 1
        degree = pol_degree
        indices = np.linspace(0, n-1, degree+2, dtype=int)
        control_points = [x_points[i] for i in indices]
        
        max_iterations = 20
        for iteration in range(max_iterations):
            # 2. Build Vandermonde matrix for the system of equations
            V = np.zeros((degree+2, degree+2))
            for i in range(degree+2):
                xi = control_points[i]
                for j in range(degree+1):
                    V[i, j] = xi**j
                V[i, degree+1] = (-1)**i  # Alternating sign for error term
            
            # Get function values at control points
            f_values = [np.interp(x, x_points, y_points) for x in control_points]
            
            # 3. Solve the system for polynomial coefficients and error
            try:
                solution = linalg.solve(V, f_values)
                coeffs = solution[:-1]  # Polynomial coefficients
                E = solution[-1]        # Error term
            except np.linalg.LinAlgError:
                print("Linear system could not be solved")
                break
            
            # 4. Find the extrema of the error function
            errors = []
            for x in x_points:
                p_val = sum(c * x**j for j, c in enumerate(coeffs))
                y = np.interp(x, x_points, y_points)
                errors.append((x, p_val - y))
            
            # 5. Find new control points at error extrema
            errors.sort(key=lambda e: abs(e[1]), reverse=True)
            new_control_points = [e[0] for e in errors[:degree+2]]
            
            # 6. Check for convergence
            if set(new_control_points) == set(control_points):
                break
            
            control_points = new_control_points
        
            # Return the final polynomial coefficients (highest power first)
        return np.flip(coeffs).copy()
    def least_square_fit(self, activation, B, samp_size, pol_degree):
        x_points = np.linspace(-B, B, samp_size)
        y_points = self.get_activation_points(activation, x_points)

        x_np = np.array(x_points)
        y_np = np.array(y_points)
        
        # Calculate coefficients using numpy
        if pol_degree == -1:
            pol_degree = len(x_points) - 1
        return np.polyfit(x_np, y_np, pol_degree)
        
    def forward(self, x):
        """
        Evaluate polynomial using Horner's method
        x can be a single value or a tensor of values
        """
        powers = x.unsqueeze(-1).pow(self.exponents)
        result = (self.coefficients * powers).sum(dim=-1)
        self.current_boundary_loss = self.get_boundary_loss(x)
        return result



class HEActivation(nn.Module):
    def __init__(self, activation='relu', activation_params=None):
        super().__init__()
        self.activation_type = activation
        self.activation_params = activation_params or {}
        
        if activation == 'poly':
            
            self.activation = PolyFit(
                activation=self.activation_params['ori_activation'],
                B=self.activation_params['B'],
                samp_size=self.activation_params['samp_size'],
                pol_degree=self.activation_params['pol_degree'],
                learnable_coeff=self.activation_params['learnable_coeffs'],
                initialization=self.activation_params['initialization'],
                boundary_loss_params=self.activation_params['boundary_loss_params']
            )
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.activation(x)