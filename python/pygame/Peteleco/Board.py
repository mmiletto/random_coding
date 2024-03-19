import pygame
from itertools import cycle

from Peteleco.Piece import Piece


class Board:

    board_square_colors = [(0, 0, 0), (240, 240, 240)]  # Black and grey

    p1_color = (0, 0, 255, 128)
    p2_color = (255, 0, 0, 128)

    def __init__(self, _width: int, _n_cells: int, offset: float):
        self.width = _width
        self.height = _width

        self.n_cells = _n_cells
        self.cell_size = _width / _n_cells
        self.half_cell_size = self.cell_size / 2.0

        self.offset = offset

    def draw(self, surface: pygame.surface.Surface):

        colors = cycle(self.board_square_colors)
        for i in range(0, self.n_cells):
            curr_color = next(colors)
            for j in range(0, self.n_cells):
                x = i * self.cell_size + self.offset
                y = j * self.cell_size + self.offset

                curr_color = next(colors)
                # Draw the board squares
                pygame.draw.rect(surface, curr_color, (x, y, self.cell_size, self.cell_size))

    def setup_players_pieces(self) -> list[Piece]:

        pieces = list()

        # Used to offset pieces in a row
        row_offset = 0.0

        # Player 1 pieces
        for j in [0, 1, 2]:
            row_offset = self.cell_size if j % 2 == 0 else 0.0

            for i in range(0, self.n_cells, 2):
                x = i * self.cell_size + self.half_cell_size + self.offset + row_offset
                y = j * self.cell_size + self.half_cell_size + self.offset
                pieces.append(Piece(x, y, self.p1_color, 30.0))

        # Player 2 pieces
        for j in [5, 6, 7]:
            row_offset = self.cell_size if j % 2 == 0 else 0.0

            for i in range(0, self.n_cells, 2):
                x = i * self.cell_size + self.half_cell_size + self.offset + row_offset
                y = j * self.cell_size + self.half_cell_size + self.offset
                pieces.append(Piece(x, y, self.p2_color, 30.0))

        return pieces

    def is_inside(self, center: tuple[float, float]) -> bool:
        x, y = center
        limit_x = self.n_cells * self.cell_size + self.offset
        limit_y = limit_x

        if self.offset <= x <= limit_x and self.offset <= y <= limit_y:
            return False

        return True
