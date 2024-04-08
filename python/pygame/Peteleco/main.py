import math

import pygame
import sys
from Peteleco.Board import Board
from Peteleco.Controller import Controller
from Peteleco.Piece import Piece


def get_distance_between_pieces(piece1: Piece, piece2: Piece) -> float:
    dx = piece1.x - piece2.x
    dy = piece1.y - piece2.y
    return math.sqrt(dx * dx + dy * dy)


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
    screen_width, screen_height = 1000, 1000
    pieces_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    window = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)

    pygame.display.set_caption("Peteleco Board Game")

    # Define colors
    background_color_white = (255, 255, 255)  # White

    # Board Instance
    board_width = 800
    board = Board(board_width, 8, (screen_height - board_width) / 2.0)

    # Create players pieces
    pieces = board.setup_players_pieces()

    # Create game controller
    controller = Controller()

    # drag control
    dragging = False
    mouse_pos = None
    index = 0

    # Main loop
    while True:

        # Draw the board
        board.draw(pieces_surface)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button press
                    dragging = True
                    mouse_pos = [event.pos[0], event.pos[1]]
                    index = find_nearest_piece(pieces, mouse_pos[0], mouse_pos[1])
                    controller.set_cursor_position(event.pos[0], event.pos[1])
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button release
                    dragging = False
                    mouse_pos = None
                    if not controller.is_cursor_inside_piece(pieces[index]):
                        controller.set_piece_velocity(pieces[index])
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_pos = [event.pos[0], event.pos[1]]
                    controller.set_cursor_position(event.pos[0], event.pos[1])

        if mouse_pos is not None and len(pieces) > 0:
            controller.set_piece_position(pieces[index])
            pieces[index].select()
            controller.draw(pieces_surface)
        elif len(pieces) > 0 and index < len(pieces):
            pieces[index].deselect()

        # Check for collisions
        for i, p1 in enumerate(pieces[:-1]):
            for j, p2 in enumerate(pieces[i+1:]):
                j += i+1
                distance = get_distance_between_pieces(p1, p2)

                if distance <= 2 * p1.radius:

                    # Calculate new velocities after collision
                    normal = [p2.x - p1.x, p2.y - p1.y]
                    unit_normal = [normal[0] / distance, normal[1] / distance]
                    unit_tangent = [-unit_normal[1], unit_normal[0]]

                    # Move the circles along the normal direction to avoid overlap
                    overlap = 2 * p1.radius - distance
                    p1.x -= overlap * unit_normal[0]
                    p1.y -= overlap * unit_normal[1]
                    p2.x += overlap * unit_normal[0]
                    p2.y += overlap * unit_normal[1]

                    # Resolve velocities into normal and tangent components
                    v1n = p1.velocity[0] * unit_normal[0] + p1.velocity[1] * unit_normal[1]
                    v1t = p1.velocity[0] * unit_tangent[0] + p1.velocity[1] * unit_tangent[1]
                    v2n = p2.velocity[0] * unit_normal[0] + p2.velocity[1] * unit_normal[1]
                    v2t = p2.velocity[0] * unit_tangent[0] + p2.velocity[1] * unit_tangent[1]

                    # Calculate new normal velocities after collision
                    new_v1n = (v1n * (p1.mass - p2.mass) + 2 * p2.mass * v2n) / (p1.mass + p2.mass)
                    new_v2n = (v2n * (p1.mass - p2.mass) + 2 * p1.mass * v1n) / (p1.mass + p2.mass)

                    # Combine normal and tangent components to get new velocities
                    new_vel1 = [new_v1n * unit_normal[0] + v1t * unit_tangent[0],
                                new_v1n * unit_normal[1] + v1t * unit_tangent[1]]
                    new_vel2 = [new_v2n * unit_normal[0] + v2t * unit_tangent[0],
                                new_v2n * unit_normal[1] + v2t * unit_tangent[1]]

                    print(f"collision between {i} and {j}")
                    pieces[i].set_velocity(new_vel1[0], new_vel1[1])
                    pieces[j].set_velocity(new_vel2[0], new_vel2[1])

        # Move pieces with updated velocities
        for piece in pieces:
            piece.move()
            piece.update_velocity()

        # Check for pieces that have moved outside the board
        pieces = [p for p in pieces if not board.is_inside(p.get_center())]

        for piece in pieces:
            piece.draw(pieces_surface)

        # Update the display
        # Fill the canvas with the background color
        window.fill(background_color_white)
        window.blit(pieces_surface, (0, 0, 0, 0))

        pygame.display.flip()
        # todo update can be used to redraw only some parts of the screen. Like we dont need to redraw the board
        # pygame.display.update()


if __name__ == '__main__':
    main()
