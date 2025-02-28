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
                piece_image = pygame.transform.scale(piece_image,
                                                     (SQUARE_SIZE, SQUARE_SIZE))  # Масштабируем изображение
                screen.blit(piece_image, (j * SQUARE_SIZE, i * SQUARE_SIZE))


def board_to_array(board):
    array = np.zeros((8, 8, 12))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        layer = piece.piece_type - 1
        color = 1 if piece.color == chess.WHITE else -1
        array[square // 8, square % 8, layer] = color

    return array


def predict_move(board):
    position_array = board_to_array(board)
    position_array = np.expand_dims(position_array, axis=0)
    prediction = model.predict(position_array)
    return prediction  # Вернуть предсказание для дальнейшей обработки


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


def transform_pawn(move, board):
    if board.piece_type_at(move.from_square) == chess.PAWN:
        if (board.color_at(move.from_square) == chess.WHITE and move.to_square // 8 == 7) or \
                (board.color_at(move.from_square) == chess.BLACK and move.to_square // 8 == 0):
            # Превращение пешки в ферзя (можно добавить выбор фигуры)
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move  # Возвращаем оригинальный ход, если превращения не происходит


def main():
    board = chess.Board()
    selected_square = None
    dragging_piece = None  # Переменная для хранения перетаскиваемой фигуры

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
                    dragging_piece = board.piece_at(selected_square)  # Сохраняем фигуру для перетаскивания

            if event.type == pygame.MOUSEBUTTONUP:
                if dragging_piece:  # Если фигура перетаскивается
                    x, y = event.pos
                    x //= SQUARE_SIZE
                    y //= SQUARE_SIZE
                    new_square = y * 8 + x

                    # Проверка на недопустимые ходы
                    if new_square != selected_square:  # Проверяем, что новая клетка отличается от оригинальной
                        move = chess.Move.from_uci(
                            f"{chess.square_name(selected_square)}{chess.square_name(new_square)}")
                        if move in board.legal_moves:
                            move = transform_pawn(move, board)  # Обработка превращения пешки
                            board.push(move)
                            dragging_piece = None  # Сбрасываем перетаскиваемую фигуру
                            selected_square = None  # Сбрасываем выделение
                            # Ход AI
                            ai_move = best_move(board, 3)
                            if ai_move:
                                board.push(ai_move)
                        else:
                            dragging_piece = None  # Сбрасываем перетаскиваемую фигуру
                            selected_square = None  # Сбрасываем выделение

            if event.type == pygame.MOUSEMOTION:
                if dragging_piece:  # Обработка перетаскивания
                    # Обновление позиции перетаскиваемой фигуры
                    screen.fill((255, 255, 255))  # Очищаем экран перед отрисовкой

                    draw_board(board)  # Отрисовываем доску
                    mouse_x, mouse_y = event.pos  # Получаем координаты мыши
                    piece_image = PIECE_IMAGES[dragging_piece.symbol()]
                    piece_image = pygame.transform.scale(piece_image,
                                                         (SQUARE_SIZE, SQUARE_SIZE))  # Масштабируем изображение
                    screen.blit(piece_image, (mouse_x - SQUARE_SIZE // 2,
                                              mouse_y - SQUARE_SIZE // 2))  # Отрисовываем фигуру под указателем мыши

        draw_board(board)
        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()