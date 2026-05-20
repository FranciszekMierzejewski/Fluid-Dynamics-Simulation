import numpy as np

class Grid:
    def __init__(self, number_of_columns: int, number_of_rows: int, grid_x_length: float = 1.0, grid_y_length: float = 1.0):
        """
        Navier Stokes describes a fluid at every point in space, hence being continuous. 
        Instead, we find average velocity and pressure in each region of grid.
        Hence, we split the grid into regions 
        
        Args:
            number_of_columns (int): Number of columns in our grid
            number_of_rows (int): Number of rows in our grid
            grid_x_length (float): Total length of x for grid
            grid_y_length (float): Total length of y for grid
        """

        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.grid_x_length = grid_x_length
        self.grid_y_length = grid_y_length

        self.cell_x_length = grid_x_length / number_of_columns
        self.cell_y_length = grid_y_length / number_of_rows

        # np.linspace(start, end (incl.), amount to split), array of even intervals
        self.cell_x = np.linspace(0.5 * self.cell_x_length, grid_x_length - 0.5 * self.cell_x_length, number_of_columns)
        self.cell_y = np.linspace(0.5 * self.cell_y_length, grid_y_length - 0.5 * self.cell_y_length, number_of_rows)

        # create a 2d grid of all combinations
        self.mesh_x, self.mesh_y = np.meshgrid(self.cell_x, self.cell_y)
    
    @property
    def grid_shape(self) -> tuple[int, int]:
        """
        Returns row x column matrix size of grid
        """
        return (self.number_of_rows, self.number_of_columns)

    @property
    def grid_length(self) -> tuple[float, float]:
        """
        Returns length of x and y of grid
        """
        return (self.grid_x_length, self.grid_y_length)

    @property
    def grid_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a mesh showing centre of each cell region.
        """
        return (self.mesh_x, self.mesh_y)
    
    @property
    def grid_ones(self) -> np.ndarray:
        """
        Does not mutate original, creates a new copy of matrix with same dimensions filled with 1s
        """
        return np.ones(self.grid_shape)
    
    @property
    def grid_zeros(self) -> np.ndarray:
        """
        Does not mutate original, creates a new copy of matrix with same dimensions filled with 0s
        """
        return np.zeros(self.grid_shape)
    
