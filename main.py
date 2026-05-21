from grid import Grid

def main():
    grid = Grid(5,7,10,10)
    print(grid.grid_shape)
    print(grid.grid_length)
    print(f"Cell x: {grid.cell_x}")
    print(f"Cell y: {grid.cell_y}")

    i, j = 1, 1
    x, y = grid.get_cell_centre(i,j)
    print(f"Cell Centre at row {i}, col {j}: ({x:.3f}, {y:.3f})")

if __name__ == "__main__":
    main()