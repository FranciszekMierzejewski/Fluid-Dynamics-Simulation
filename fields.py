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