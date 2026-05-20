from grid import Grid

def main():
    grid = Grid(2,4,10,10)
    print(grid.grid_shape)
    print(grid.grid_length)
    print(grid.grid_centres)

if __name__ == "__main__":
    main()