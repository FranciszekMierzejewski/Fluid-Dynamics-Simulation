import numpy as np
from grid import Grid

class Fields():
    def __init__(self, grid: Grid):
        """
        Stores three physical fields of the fluid, the x velocity component, y velocity component, and pressure. 
        """
        self.grid = grid
        self.x: np.ndarray = grid.grid_zeros() # x component of velocity
        self.y: np.ndarray = grid.grid_zeros() # y component of velocity
        self.p: np.ndarray = grid.grid_zeros() # pressure 

    
    def reset(self):
        self.x = self.grid.grid_zeros()
        self.y = self.grid.grid_zeros()
        self.p = self.grid.grid_zeros()

    def max_velocity(self) -> float:
        return max(np.max(np.abs(self.x)), np.max(np.abs(self.y))) # take modulus of each cell in each velocity field, take max of each field, and compare max
    
    def resultant_velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate resulatant velocity from x and y components of each cell in the velocity fields
        """
        return np.sqrt(x**2 + y**2) 

    def resultant_angle(self) -> np.ndarray:
        """
        Calculate angle of resultant velocity of each cell in the velocity fields, in radians.
        
        Domain of arctan = (-∞, ∞)
        Range of arctan = (-π/2, π/2)
        """
        return np.arctan(self.resultant_velocity(self.x, self.y)) 

    def __repr__(self) -> str:
        #return f"Fields: \nx=\n{self.x}, \ny=\n{self.y}, \np=\n{self.p}"
        return f"Resultant Field: \n{self.resultant_velocity(self.x, self.y)}\n Resultant Angle: \n {self.resultant_angle()}"
    
if __name__ == "__main__":
    grid = Grid(100,100,1.0,1.0)
    fields = Fields(grid)
    print(fields)