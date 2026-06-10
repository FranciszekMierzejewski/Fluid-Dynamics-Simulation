# Given a randomnly generated velocity field, we compute our divergence at each cell, solve for pressure and project velocity field to be divergence-free, hence allowing for incompressible flow.

import numpy as np
from grid import Grid
from fields import Fields

class PressureProjection():
    def __init__(self, fields: Fields):
        self.fields = fields
        self.grid = fields.grid

    def calculate_divergence(self) -> np.ndarray:
        return self.grid.divergence(self.fields.x, self.fields.y)

    def calculate_pressure(self):
        """
        ∇²p = (ρ/dt) ∇·u
        x_i^(k+1) = x_i^(k) + ω((x_i^(k+1))_GS - x_i^(k))
        I will use Successive Over-Relexation (1 < ω < 2), a variant of the Gauss-Seidel method, which computes a weighted average of previous and new iterate, allowing for faster convergence.
        
        For N^2 grid: 
          ω_opt ~~ 2/(1 + (1 - cos(π/N)^1/2) = 2/(1 + sin(π/N))
        """

        # for each iteration, go across each cell in roster order and update pressure 
        max_mean = 1/2 * (np.cos(np.pi/self.grid.number_of_rows) + np.cos(np.pi/self.grid.number_of_columns))
        optimal_weight = 2/(1 + (1 - max_mean)** 0.5)
    

"""
import numpy as np
def SOR(A, b, x, w):
  A = np.array(A); b = np.array(b); x = np.array(x);
  n = len(A)
  xnew = np.zeros(n)
  for i in range(n):
    xnew[i] = (1 - w)*x[i] + w*(b[i] - sum([A[i, j]*x[j] for j in range(i + 1, n)]) - sum([A[i, j]*xnew[j] for j in range(i)]))/A[i, i]
  return xnew
  
A = [[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]]
b = [7.85, -19.3, 71.4]
x = [[1, 1, 1.]]
MaxIter = 100
ErrorTable = [1]
eps = 0.001
i = 1
omega = 1.1

while i <= MaxIter and abs(ErrorTable[i - 1]) > eps: 
  xi = SOR(A, b, x[i - 1],omega)
  x.append(xi)
  ei = np.linalg.norm(x[i] - x[i - 1])/np.linalg.norm(x[i]) 
  ErrorTable.append(ei) 
  i+=1
print("x:",np.array(x))
print("ErrorTable:",np.vstack(ErrorTable))
"""