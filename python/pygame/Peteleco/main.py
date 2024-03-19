import math

import pygame
import sys
from Peteleco.Board import Board
from Peteleco.Piece import Piece


def get_distance(piece: Piece, x: float, y: float) -> float:
    dx = piece.x - x
    dy = piece.y - y
    return math.sqrt(dx * dx + dy * dy)


def find_nearest_piece(pieces: list[Piece], x: float, y: float):

    min_dist = float("inf")
    index = -1

    for i, piece in enumerate(pieces):
        dist = get_distance(piece, x, y)
        if dist < min_dist:
            min_dist = dist
            index = i

    return index


def main():
    # Initialize Pygame
    pygame.init()

    # Set up the screen
    screen_width, screen_height = 900, 900
    canvas = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    window = pygame.display.set_mode((screen_width, screen_height))

    pygame.display.set_caption("Peteleco Board Game")

    # Define colors
    background_color_white = (255, 255, 255)  # White

    # Board Instance
    board_width = 800
    board = Board(board_width, 8, (screen_height - board_width) / 2.0)

    # Create players pieces
    pieces = board.setup_players_pieces()

    # drag control
    dragging = False
    mouse_pos = None
    index = 0

    # Main loop
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    dragging = True
                    mouse_pos = [event.pos[0], event.pos[1]]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
                    mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_pos = [event.pos[0], event.pos[1]]

        if mouse_pos is not None and len(pieces) > 0:
            index = find_nearest_piece(pieces, mouse_pos[0], mouse_pos[1])
            pieces[index].set_center(mouse_pos[0], mouse_pos[1])
            pieces[index].select()
        elif len(pieces) > 0:
            pieces[index].deselect()

        # Fill the canvas with the background color
        window.fill(background_color_white)

        # Draw the board
        board.draw(canvas)

        # Check for pieces that have moved outside the board
        pieces = [p for p in pieces if not board.is_inside(p.get_center())]

        for piece in pieces:
            piece.draw(canvas)

        # Update the display
        window.blit(canvas, (0, 0, 0, 0))
        pygame.display.flip()
        # todo update can be used to redraw only some parts of the screen. Like we dont need to redraw the board
        # pygame.display.update()


if __name__ == '__main__':
    main()
