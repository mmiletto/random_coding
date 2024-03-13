import sys

from Grid2D import *
from WindowPyGame import *

WIN_WIDTH_PIXELS = 700
WIN_HEIGHT_PIXELS = 700

if __name__ == '__main__':

    # setup domain configuration
    grid = Automata2DGrid(30, 30)
    win = WindowPyGame(WIN_WIDTH_PIXELS, WIN_HEIGHT_PIXELS, "Fluid simulation Cellular Automata", grid.n_cells_x,
                       grid.n_cells_y, 1)
    grid.set_cells_width(WIN_WIDTH_PIXELS/grid.n_cells_x, WIN_HEIGHT_PIXELS/grid.n_cells_y)

    # setup fluid cells
    for x in range(3, 7):
        for y in range(3, 5):
            grid.add_fluid_cell(x, y)

    running = True

    # main loop
    while running:

        win.draw()

        win.draw_cells(grid.get_cells())

        win.draw_grid(grid.n_cells_x, grid.n_cells_y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        win.finish_events()

        grid.compute()

    # Quit Pygame properly
    pygame.quit()
    sys.exit()
