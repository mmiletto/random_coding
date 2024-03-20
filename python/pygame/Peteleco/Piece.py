import pygame


class Piece:

    def __init__(self, x: float, y: float, color: tuple[int, int, int, int], radius: float):
        self.x = x
        self.y = y
        self.radius = radius

        self.color = color

        self.velocity = (0.0, 0.0)
        self.friction_factor = 0.01

        self.mass = 10.0

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

    def move(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def update_velocity(self):
        ux, uy = self.velocity
        ux *= (1.0 - self.friction_factor)
        uy *= (1.0 - self.friction_factor)
        self.velocity = (ux, uy)

    def set_velocity(self, x: float, y: float) -> None:
        self.velocity = (x, y)
