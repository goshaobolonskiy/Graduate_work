import chess
import pygame
import sys
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('chess_ai_model.h5')

# Определяем размеры доски и цветовые схемы
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
COLORS = [(238, 238, 210), (118, 150, 86)]

# Загрузка изображений фигур
PIECE_IMAGES = {
    'P': pygame.image.load('images/white_pawn.png'),
    'p': pygame.image.load('images/black_pawn.png'),
    'N': pygame.image.load('images/white_knight.png'),
    'n': pygame.image.load('images/black_knight.png'),
    'B': pygame.image.load('images/white_bishop.png'),
    'b': pygame.image.load('images/black_bishop.png'),
    'R': pygame.image.load('images/white_rook.png'),
    'r': pygame.image.load('images/black_rook.png'),
    'Q': pygame.image.load('images/white_queen.png'),
    'q': pygame.image.load('images/black_queen.png'),
    'K': pygame.image.load('images/white_king.png'),
    'k': pygame.image.load('images/black_king.png'),
}

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Шахматы против НС")
clock = pygame.time.Clock()

def draw_board(board):
    for i in range(8):
        for j in range(8):
            color = COLORS[(i + j) % 2]
            pygame.draw.rect(screen, color, (j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(i * 8 + j)
            if piece:
                piece_image = PIECE_IMAGES[piece.symbol()]
                piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(piece_image, (j * SQUARE_SIZE, i * SQUARE_SIZE))

def transform_pawn(move, board):
    # Проверка, является ли текущий ход превращением пешки
    if board.piece_type_at(move.from_square) == chess.PAWN:
        if (board.color_at(move.from_square) == chess.WHITE and move.to_square // 8 == 7) or \
           (board.color_at(move.from_square) == chess.BLACK and move.to_square // 8 == 0):
            # Превращение пешки в ферзя
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move  # Если превращения не происходит, возвращаем оригинальный ход

def evaluate_board(board):
    material_count = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    score = 0
    for piece_type in material_count.keys():
        score += len(board.pieces(piece_type, chess.WHITE)) * material_count[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * material_count[piece_type]

    return score

def minimax(board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

def best_move(board, depth):
    best_eval = float('-inf')
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, False)
        board.pop()

        if eval > best_eval:
            best_eval = eval
            best_move = move

    return best_move

def main():
    board = chess.Board()
    selected_square = None
    dragging_piece = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x //= SQUARE_SIZE
                y //= SQUARE_SIZE

                if board.color_at(y * 8 + x) == chess.WHITE:  # Проверяем, что выбрана белая фигура
                    selected_square = y * 8 + x
                    dragging_piece = board.piece_at(selected_square)

            if event.type == pygame.MOUSEBUTTONUP:
                if dragging_piece:  # Если фигура перетаскивается
                    x, y = event.pos
                    x //= SQUARE_SIZE
                    y //= SQUARE_SIZE
                    new_square = y * 8 + x

                    # Проверяем, отличается ли новая клетка от старой
                    if new_square != selected_square:
                        move = chess.Move.from_uci(
                            f"{chess.square_name(selected_square)}{chess.square_name(new_square)}")
                        # Обработка превращения пешки
                        move = transform_pawn(move, board)
                        if move in board.legal_moves:
                            print(move)
                            board.push(move)  # Выполняем ход на доске
                            dragging_piece = None
                            selected_square = None

                            # Ход AI
                            ai_move = best_move(board, 3)
                            if ai_move:
                                board.push(ai_move)
                        else:
                            dragging_piece = None
                            selected_square = None

            if event.type == pygame.MOUSEMOTION:
                if dragging_piece:  # Обработка перетаскивания
                    screen.fill((255, 255, 255))
                    draw_board(board)  # Отрисовываем доску
                    mouse_x, mouse_y = event.pos
                    piece_image = PIECE_IMAGES[dragging_piece.symbol()]
                    piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                    screen.blit(piece_image, (mouse_x - SQUARE_SIZE // 2, mouse_y - SQUARE_SIZE // 2))

        draw_board(board)
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()