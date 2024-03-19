import pygame


class Piece:

    def __init__(self, x: float, y: float, color: tuple[int, int, int, int], radius: float):
        self.x = x
        self.y = y
        self.radius = radius

        self.color = color

        self.velocity = (0.0, 0.0)
        self.friction_factor = 0.4

    def draw(self, surface: pygame.surface.Surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

    def get_center(self) -> (float, float):
        return self.x, self.y

    def set_center(self, x: float, y: float):
        self.x = x
        self.y = y

    def select(self):
        r, g, b, a, = self.color
        self.color = (r, g, b, 255)

    def deselect(self):
        r, g, b, a, = self.color
        self.color = (r, g, b, 128)
