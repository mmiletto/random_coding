import pygame


class WindowPyGame:

    def __init__(self, _width, _height, window_title, n_cells_x, n_cells_y, target_fps=60):
        # Initialize Pygame
        pygame.init()

        self.width = _width
        self.height = _height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(window_title)

        self.fps = target_fps
        self.frame_index = 0
        self.clock = pygame.time.Clock()

        # cell size in screen pixels
        self.cell_size_x = self.width / n_cells_x
        self.cell_size_y = self.height / n_cells_y

    def draw(self):
        # Set up the colors
        black = (0, 0, 0)
        # Draw the background
        self.screen.fill(black)

    def draw_grid(self, n_cells_x, n_cells_y):
        white = (255, 255, 255)

        vertical_lines = [x * self.cell_size_x for x in range(n_cells_x)]
        horizontal_lines = [y * self.cell_size_y for y in range(n_cells_y)]

        for vl in vertical_lines:
            # Parameters: surface, color, start point, end point, line thickness
            pygame.draw.line(self.screen, white, (vl, 0), (vl, self.height), 1)

        for hl in horizontal_lines:
            # Parameters: surface, color, start point, end point, line thickness
            pygame.draw.line(self.screen, white, (0, hl), (self.width, hl), 1)

    def draw_cells(self, cell_list):
        blue = (0, 0, 255)

        for x, y in cell_list:
            pygame.draw.rect(self.screen, blue, (x, y, self.cell_size_x, self.cell_size_y))

    def finish_events(self):
        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)
