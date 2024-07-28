class Automata2DGrid:
    def __init__(self, n_cells_x: int, n_cells_y: int):

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.cell_width = 0
        self.cell_height = 0

        # all grid cells here
        self.grid_values = [[0] * self.n_cells_y] * self.n_cells_x

        self.fluids_cells_list = []

    def add_fluid_cell(self, i, j):
        self.grid_values[i][j] = 1
        self.fluids_cells_list.append(self.get_cell_center_coord(i, j))

    def get_cell(self, i, j):
        return self.grid_values[i][j]

    def get_cell_center_coord(self, i, j):
        return i * self.cell_width, j * self.cell_height

    def get_cells(self):
        return self.fluids_cells_list

    def set_cells_width(self, cell_width_pixels, cell_height_pixels):
        self.cell_width = cell_width_pixels
        self.cell_height = cell_height_pixels

    def compute(self):
        for i in range(0, len(self.fluids_cells_list)):
            cell = self.fluids_cells_list[i]
            self.fluids_cells_list[i] = (cell[0], cell[1] + self.cell_height)