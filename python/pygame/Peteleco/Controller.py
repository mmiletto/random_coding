import math

import pygame

from Peteleco.Piece import Piece


class Controller:

    def __init__(self):
        self.cursor_x = None
        self.cursor_y = None

        self.piece_x = None
        self.piece_y = None

    def set_cursor_position(self, x: float, y: float) -> None:
        self.cursor_x = x
        self.cursor_y = y

    def set_piece_position(self, piece: Piece) -> None:
        self.piece_x = piece.x
        self.piece_y = piece.y

    def is_cursor_inside_piece(self, piece: Piece) -> bool:

        x, y = piece.get_center()
        dist = abs(self.cursor_x - x) + abs(self.cursor_y - y)
        if dist <= piece.radius:
            return True
        return False

    def draw(self, surface: pygame.surface.Surface):
        pygame.draw.line(surface, (0, 255, 0), (self.piece_x, self.piece_y), (self.cursor_x, self.cursor_y), width=5)

    def set_piece_velocity(self, piece: Piece):
        ux = (piece.x - self.cursor_x) * 1.5
        uy = (piece.y - self.cursor_y) * 1.5

        max_force = 50.0
        vector_len = math.sqrt(ux * ux + uy * uy)
        vector_len = max_force if vector_len > max_force else vector_len

        base_force = 0.0005
        piece.set_velocity(ux * base_force * vector_len, uy * base_force * vector_len)
