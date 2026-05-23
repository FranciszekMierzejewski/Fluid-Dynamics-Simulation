from grid import Grid
import numpy as np

def main():
    grid = Grid(100,100,1.0,1.0)

    # Properties
    print(f"Shape: {grid.grid_shape}") # expect (100,100)
    print(f"Length: {grid.grid_length}") # expect (1.0, 1.0)
    print(f"Cell Size: {grid.cell_x_length:.3f},{grid.cell_y_length:.3f}") # expect (0.010, 0.010)
    
    i, j = 1, 1
    x, y = grid.get_cell_centre(i,j)
    print(f"Cell Centre at row {i}, col {j}: ({x:.3f}, {y:.3f})")

    
    # Central Difference 
        # Testing f = x^2, therefore expected ∂f/∂x = 2x
        # Testing f = y^2, therefore expected ∂f/∂y = 2y
    
    f = grid.mesh_x ** 2
    df_dx = grid.central_difference_x(f)
    expected_df_dx = 2 * grid.mesh_x
    error_x = np.max(np.abs(df_dx[2:-2,2:-2] - expected_df_dx[2:-2,2:-2]))
    print(f"Maximum error in centre regions: {error_x:.6f}")

    f = grid.mesh_y ** 2
    df_dy = grid.central_difference_y(f)
    expected_df_dy = 2 * grid.mesh_y
    error_y = np.max(np.abs(df_dy[2:-2,2:-2] - expected_df_dy[2:-2,2:-2]))
    print(f"Maximum error in centre regions: {error_y:.6f}")

    # Gradient
        # Testing f = x^2 + y^2
        # Expected: ∇f = (2x, 2y)
    
    f = grid.mesh_x ** 2 + grid.mesh_y ** 2
    gradient_x, gradient_y = grid.gradient(f)
    
    print(f"∂f/∂x max error: {np.max(np.abs(gradient_x[2:-2,2:-2] - 2*grid.mesh_x[2:-2,2:-2])):.2e}")
    print(f"∂f/∂y max error: {np.max(np.abs(gradient_y[2:-2,2:-2] - 2*grid.mesh_y[2:-2,2:-2])):.2e}")

    # Divergence
        # Testing u=x, v=y
        # Expect: ∇·(u,v) = ∂x/∂x + ∂y/∂y=1+1=2 for all cell regions
    
    u = grid.mesh_x
    v = grid.mesh_y
    divergence = grid.divergence(u, v)
    print(f"Max error from 2.0: {np.max(np.abs(divergence[2:-2,2:-2] - 2.0)):.2e}")

if __name__ == "__main__":
    main()