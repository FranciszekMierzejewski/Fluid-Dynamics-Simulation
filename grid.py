import numpy as np

class Grid:
    def __init__(self, number_of_columns: int, number_of_rows: int, grid_x_length: float = 1.0, grid_y_length: float = 1.0):
        """
        Navier Stokes equations describe fluid motion using continuous fluids for velocity and pressure
        We discretise the fluid domain into grid cells, and approximate the centre of these fields

        # ∇·u = 0 assumed
            # Describing a velocity field u where fluid is incompressible
            # ∂ρ/∂t + ∇·(uρ) is reduced given constant density. Local volume is conserved
        
        # ρ(DV->)/Dt = -∇p + ρ(g)-> + μ∇^2(V)->
            # where: 
                # ρ(DV->)/Dt = ρ[∂V/∂t + (V·∇)V])], is the inertia of the fluid
                # -∇p describes the fluid flowing in direction of largest change in pressure
                # ρ(g)-> is the hydrostatic weight density, describing external forces acting on the fluid (gravitational or electromagnetic)
                # μ∇^2(V)-> is the diffusion term, where for a Newtonian fluid, viscosity operates as a diffusion of momentum. Viscous friction
        
        # Substituting:
            # ρ[∂V/∂t + (V·∇)V] = -∇p + ρ(g)-> + μ∇^2(V)->
        # Rearranging:
            # ∂V/∂t = -(V·∇)V - ∇p/ρ + g-> + (μ/ρ)∇^2(V)->
        # Since v = μ/ρ:
            # ∂V/∂t = -(V·∇)V - ∇p/ρ + g-> + (v)∇^2(V)->
        
        # Which is the equation we shall use
        
        Args:
            number_of_columns (int): Number of columns in our grid
            number_of_rows (int): Number of rows in our grid
            grid_x_length (float): Total length of x for grid
            grid_y_length (float): Total length of y for grid
        """

        if number_of_columns < 5 or number_of_rows < 5:
            raise ValueError("Must be at least 5 columns and 5 rows for grid")
            # f(x-2h), f(x-h), f(x), f(x+h), f(x+2h) for 4th order central difference
        
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

        self.centre_values: np.ndarray = self.grid_zeros()
    
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
            return float(self.centre_values[i, j])
        raise ValueError(f"Please enter coordinates between 0 and {self.number_of_rows-1} inclusive for i and between 0 and {self.number_of_columns-1} inclusive for j")
    
    def set_value(self, i:int, j:int, value: float) -> None:
        if self.boundary_cell_check(i, j):
            self.centre_values[i, j] = value
            return
        raise ValueError(f"Please enter coordinates between 0 and {self.number_of_rows-1} inclusive for i and between 0 and {self.number_of_columns-1} inclusive for j")
    
    def first_diff_x_central_difference(self, f: np.ndarray) -> np.ndarray:
        """
        Returns central difference to approximate the derivative of function, by sampling values on both sides of point 

        # ∇ · F ~~ (u[i+1,j] - u[i-1,j])/(2*dx) + (v[i,j+1] - v[i,j-1])/(2*dy) : 2nd order  
        # f'(x) ~~ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h) : 4th order  
        """

        difference_grid_x: np.ndarray = self.grid_zeros()
        dx: float = self.cell_x_length
 
        difference_grid_x[:,2:-2] = (f[:,:-4] - 8*f[:,1:-3] + 8*f[:,3:-1] - f[:,4:]) / (12 * dx) # all rows, third column to third to last column
        difference_grid_x[:,1] = (f[:,2] - f[:,0]) / (2 * dx) # all rows, second column
        difference_grid_x[:,-2] = (f[:,-1] - f[:,-3]) / (2 * dx) # all rows, second to last column

        return difference_grid_x
    
    def first_diff_y_central_difference(self, f: np.ndarray) -> np.ndarray:
        difference_grid_y: np.ndarray = self.grid_zeros()
        dy: float = self.cell_y_length

        difference_grid_y[2:-2,:] = (f[:-4,:] - 8*f[1:-3,:] + 8*f[3:-1,:] - f[4:,:]) / (12 * dy) # third row to third to last row, all columns
        difference_grid_y[1,:] = (f[2,:] - f[0,:]) / (2 * dy) # second row, all columns
        difference_grid_y[-2,:] = (f[-1,:] - f[-3,:]) / (2 * dy) # second to last row, all columns

        return difference_grid_y
    
    def second_diff_x_central_difference(self, f: np.ndarray) -> np.ndarray:
        """
        f''(x) ~~ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h²) : 4th order 
        """
        difference_grid_x: np.ndarray = self.grid_zeros()
        dx_squared: float = self.cell_x_length ** 2

        difference_grid_x[:,2:-2] = (-f[:,:-4] + 16*f[:,1:-3] - 30*f[:,2:-2] + 16*f[:,3:-1] - f[:,4:]) / (12 * dx_squared) # all rows, third column to third to last column
        difference_grid_x[:,1] = (f[:,0] - 2*f[:,1] + f[:,2]) / dx_squared # all rows, second column
        difference_grid_x[:,-2] = (f[:,-3] - 2*f[:,-2] + f[:,-1]) / dx_squared # all rows, second to last column

        return difference_grid_x

    def second_diff_y_central_difference(self, f: np.ndarray) -> np.ndarray:
        difference_grid_y: np.ndarray = self.grid_zeros()
        dy_squared: float = self.cell_y_length ** 2

        difference_grid_y[2:-2,:] = (-f[:-4,:] + 16*f[1:-3,:] - 30*f[2:-2,:] + 16*f[3:-1,:] - f[4:,:]) / (12 * dy_squared) # third row to third to last row, all columns
        difference_grid_y[1,:] = (f[0,:] - 2*f[1,:] + f[2,:]) / dy_squared # second row, all columns
        difference_grid_y[-2,:] = (f[-3,:] - 2*f[-2,:] + f[-1,:]) / dy_squared # second to last row, all columns

        return difference_grid_y

    def gradient(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Points in direction of greatest change of function, describing pressure forces on fluid
        """
        return self.first_diff_x_central_difference(f), self.first_diff_y_central_difference(f)

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Measures how much a fluid spreads out from a point, describing the incompressibility condition of the fluid
        In our case, we want incompressible flow, so the divergence should be 0 everywhere
        ∇·F = ∂u/∂x + ∂v/∂y
        """
        return self.first_diff_x_central_difference(u) + self.first_diff_y_central_difference(v)

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        How much a value differs from its neighbours on average, describing momentum diffusion in our fluid
        ∇^2u = ∂^2u/∂x^2 + ∂^2u/∂y^2
        """
        return self.second_diff_x_central_difference(u) + self.second_diff_y_central_difference(u)



# forward difference, second order accurate:
# f'(x) ~~ (-3f(x1) + 4f(x2) - f(x3)) / (2h)

# backward difference:
# f'(x) ~~ (3f(xn) - 4f(xn-1) + f(xn-2)) / (2h)