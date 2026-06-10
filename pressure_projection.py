# Given a randomnly generated velocity field, we compute our divergence at each cell, solve for pressure and project velocity field to be divergence-free, hence allowing for incompressible flow.

import numpy as np
from grid import Grid
from fields import Fields

class PressureProjection():
    def __init__(self, fields: Fields, omega, tolerance: float = 1e-6, max_iterations: int = 10000):
        self.fields = fields
        self.grid = fields.grid
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        max_mean: float = 1/2 * (np.cos(np.pi/self.grid.number_of_rows) + np.cos(np.pi/self.grid.number_of_columns))
        self.omega: float = 2/(1 + (1 - max_mean)** 0.5)


    def _calculate_divergence(self) -> np.ndarray:
        return self.grid.divergence(self.fields.x, self.fields.y)
    
    def _calculate_source_term(self, dt: float) -> np.ndarray:
        return self._calculate_divergence() / dt
    
    def _calculate_successive_over_relaxation(self, b: np.ndarray):
        if self.fields.p is None:
            raise ValueError("Pressure field is not initialized")
        
        p = self.fields.p
        dx = (self.grid.cell_x_length, 1.0)[self.grid.cell_x_length is None]
        dy = (self.grid.cell_y_length, 1.0)[self.grid.cell_y_length is None]
        dx_squared = dx ** 2
        dy_squared = dy ** 2
        denominator = 2 * (dx_squared + dy_squared)

        for iteration in range(self.max_iterations):
            p_old = p.copy()

            for i in range(1, self.grid.grid_shape[0] - 1):
                for j in range(1, self.grid.grid_shape[1] - 1):
                    p[i, j] = (
                      self.omega * 
                      ((p[i+1, j] + p[i-1, j]) * dy_squared + 
                      (p[i, j+1] + p[i, j-1]) * dx_squared - 
                      b[i, j] * dx_squared * dy_squared) / denominator + 
                      (1 - self.omega) * p[i, j]
                    )

            residual = np.linalg.norm(p - p_old)/np.linalg.norm(p)
            
            if residual < self.tolerance:
                print(f"Successive over relaxation with weight {self.omega:.4f} has converged in {iteration + 1} iterations with residual {residual:.2e}")
                break
        print(f"Reached maximum iterations {self.max_iterations}, failed to converge")
            
        return p

