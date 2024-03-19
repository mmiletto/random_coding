import pygame
import sys


def run_simple_anim():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simple Animation")

    # Set up the colors
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Set up the initial position
    x, y = 100, 100

    # Set up the clock to control the frame rate
    clock = pygame.time.Clock()

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update the position
        x += 5
        y += 2

        # Draw the background
        screen.fill(black)

        # Draw a rectangle at the updated position
        pygame.draw.rect(screen, white, (x, y, 50, 50))

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(60)  # 30 frames per second
