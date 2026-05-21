import numpy as np

class Grid:
    def __init__(self, number_of_columns: int, number_of_rows: int, grid_x_length: float = 1.0, grid_y_length: float = 1.0):
        """
        Navier Stokes describes a fluid at every point in space, hence being continuous
        Instead, we find average velocity and pressure in each region of grid
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

        self.centre_values: dict[list[float], float] = {} 

    
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
        Returns a mesh showing centre of each cell region
        """
        return (self.mesh_x, self.mesh_y)

    def boundary_cell_check(self, i: int, j: int) -> bool:
        """
        Checks if inputs i and j are legal coordinates that lie in grid
        """
        if i < 0 or i > self.number_of_rows - 1:
            return False
        if j < 0 or j > self.number_of_columns - 1:
            return False
        return True
    
    def get_cell_centre(self, i: int, j: int) -> list[float]:
        """
        Returns the coordinates of centre of the cell at row i and column j in mesh grid
        """
        if self.boundary_cell_check(i, j):
            x = float(self.grid_centres[0][i, j])
            y = float(self.grid_centres[1][i, j])
            return [x,y]
        raise ValueError(f"Please enter coordinates between 0 and {self.number_of_rows-1} inclusive for i and between 0 and {self.number_of_columns-1} inclusive for j")
    
    def grid_ones(self) -> np.ndarray:
        """
        Does not mutate original, creates a new copy of matrix with same dimensions filled with 1s
        """
        return np.ones(self.grid_shape)
    
    def grid_zeros(self) -> np.ndarray:
        """
        Does not mutate original, creates a new copy of matrix with same dimensions filled with 0s
        """
        return np.zeros(self.grid_shape)

    def get_value(self, i:int, j:int) -> float:
        if self.boundary_cell_check(i, j):
            key = self.get_cell_centre(i,j)
            if key in self.centre_values:
                return self.centre_values[key]
            raise KeyError(f"No such key present in dictionary")
        raise ValueError(f"Please enter coordinates between 0 and {self.number_of_rows-1} inclusive for i and between 0 and {self.number_of_columns-1} inclusive for j")
    
    def set_value(self, i:int, j:int, value: float) -> None:
        if self.boundary_cell_check(i, j):
            self.centre_values[self.get_cell_centre(i,j)] = value
            return
        raise ValueError(f"Please enter coordinates between 0 and {self.number_of_rows-1} inclusive for i and between 0 and {self.number_of_columns-1} inclusive for j")
    
    def _central_difference(self, i: int, j: int, step_size: int):
        #Returns central difference to approximate the derivative of function, by sampling values on both sides of point 
        # step_size = self.cell_x_length
        #derivative: float = f(x+h) - f(x-h) / (2 * step_size)
        pass

