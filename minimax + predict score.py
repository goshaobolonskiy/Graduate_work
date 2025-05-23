import asyncio

import chess
import chess.pgn
import numpy as np
import random

# Для нейросети (можете активировать, когда нужно)
import torch
import torch.nn as nn
import torch.optim as optim

# =======================
# 1. Загрузка партий из PGN
# =======================

import pygame

pygame.init()

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
SQUARE_SIZE = 60
WIDTH, HEIGHT = 8 * SQUARE_SIZE, 8 * SQUARE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Visualization")


import matplotlib.pyplot as plt
import chess


def draw_board(board, selected_square=None, valid_moves=[]):
    # Рисуем фон квадратиков
    colors = ["#f0d9b5", "#b58863"]
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

            # Рисуем фигуру, если есть
            square = chess.square(c, 7 - r)
            piece = board.piece_at(square)
            if piece:
                img = PIECE_IMAGES[piece.symbol()]
                img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(img, (c * SQUARE_SIZE, r * SQUARE_SIZE))

    # Выделение выбранной клетки
    if selected_square is not None:
        c = selected_square % 8
        r = 7 - (selected_square // 8)
        rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, (0, 255, 0), rect, 3)

    # Выделение допустимых ходов
    for move in valid_moves:
        c = move.to_square % 8
        r = 7 - (move.to_square // 8)
        rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, (255, 255, 0), rect, 3)

    pygame.display.flip()


def load_games(pgn_file, limit=1000):
    games = []
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while limit > 0:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
            limit -= 1
    return games

# =======================
# 2. Генерация позиций
# =======================

def extract_positions(games, max_moves=60):
    positions = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            if board.fullmove_number > max_moves:
                break
            # Опционально, можно сохранять результат
            positions.append((board.copy(), game.headers.get("Result")))
    return positions

# =======================
# 3. Функция оценки
# =======================

# Таблицы для оценки фигур (примерные, можно искать готовые таблицы)
# Таблицы для открытой позиции (начальная)
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5, -10, 0, 0, -10, -5,  5,
     0, 0, 0, 20, 20, 0, 0,  0,
     5, 5, 10, 25, 25, 10, 5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PAWN_TABLE_BLACK = PAWN_TABLE[::-1]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
KNIGHT_TABLE_BLACK = KNIGHT_TABLE[::-1]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
BISHOP_TABLE_BLACK = BISHOP_TABLE[::-1]

ROOK_TABLE = [
     0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
     5, 10, 10, 10, 10, 10, 10, 5,
     0, 0, 0, 0, 0, 0, 0, 0,
]

# Таблица для ферзя
QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
QUEEN_TABLE_BLACK = QUEEN_TABLE[::-1]

# Таблица для короля в средней игре
KING_TABLE_MID = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,  0,  0,  0,  0, 20, 20,
     20,  30,  10, 0,  0, 10, 30, 20,
]

# Таблица для короля в эндшпиле
KING_TABLE_END = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

# Полные таблицы для фигур (позиционная оценка)
# Все таблицы для фигур
piece_square_tables = {
    'P': PAWN_TABLE,
    'p': PAWN_TABLE_BLACK,
    'N': KNIGHT_TABLE,
    'n': KNIGHT_TABLE_BLACK,
    'B': BISHOP_TABLE,
    'b': BISHOP_TABLE_BLACK,
    'R': ROOK_TABLE,
    'r': ROOK_TABLE,
    'Q': QUEEN_TABLE,
    'q': QUEEN_TABLE,
    'K': KING_TABLE_MID,
    'k': KING_TABLE_MID,
}

def evaluate_board(board, stage='middle'):
    """
    Оценивает текущую позицию на основе материала и позиции фигур.
    stage: 'opening', 'middle', или 'end' — стадия игры.
    """
    # Выбор таблицы для короля в зависимости от стадии
    if stage == 'end':
        king_table = KING_TABLE_END
    else:
        king_table = KING_TABLE_MID

    # Материаловые стоимости фигур
    material_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }

    score = 0
    positional_bonus = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            value = material_values[symbol]
            score += value  # материал

            # Получаем таблицу позиции для фигуры
            table = piece_square_tables.get(symbol)
            if table:
                # Индекс таблицы
                bonus = table[square]
                # Если фигура черная — зеркалим индекс для правильной оценки
                if piece.color == chess.BLACK:
                    bonus = table[square]
                # В сумме добавляем/вычитаем позиционный бонус
                if piece.color == chess.WHITE:
                    positional_bonus += bonus
                else:
                    positional_bonus -= bonus

            # Для короля используем отдельную таблицу
            if symbol in ['K', 'k']:
                bonus = king_table[square]
                if piece.color == chess.WHITE:
                    positional_bonus += bonus
                else:
                    positional_bonus -= bonus

    # Итоговая оценка с учетом майнингового веса позиционных бонусов
    total_score = score + 0.1 * positional_bonus
    return total_score


# Предположим, что у вас есть глобально загруженная модель
from tensorflow.keras.models import load_model

model = load_model('chess_eval_model.keras')


def evaluate_board_with_nn(board, stage='middle'):
    """
    Оценивает текущую позицию с помощью нейросети.
    В качестве альтернативы или дополнения можете использовать старую функцию.
    """
    def position_to_tensor(board):
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                plane = piece_to_plane[piece.piece_type]
                if piece.color == chess.WHITE:
                    tensor[row, col, plane] = 1
                else:
                    tensor[row, col, 6 + plane] = 1
        return tensor

    # преобразуем позицию в тензор
    tensor_input = position_to_tensor(board).reshape(1, 8, 8, 12)

    # получаем предсказание модели
    prediction = model.predict(tensor_input)[0][0]  # один вывод

    # можем использовать также классическую evaluate_board для сравнения
    # классическая оценка:
    # classical_score = evaluate_board(board, stage)

    # Вернем либо нейросетевую, либо классическую, по желанию
    # Например, отдаем полностью нейросеть:
    # print(prediction)
    return prediction

    # Или, например, взвешенную комбинацию:
    # return 0.8 * prediction + 0.2 * classical_score


# =======================
# 4. Минимакс с альфа-бета
# =======================

def select_move(board, depth=4):
    """
    Выбирает лучший ход с помощью minimax + альфа-бета, где оценка берется из нейросети.
    """

    def evaluate_position(board):
        return evaluate_board_with_nn(board)

    def alphabeta(board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return evaluate_position(board), None

        best_move = None
        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval, _ = alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval, _ = alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    # Запуск поиска с использованием нейросети для оценки
    _, best_move = alphabeta(board, depth, -float('inf'), float('inf'), board.turn == chess.WHITE)
    return best_move

async def async_select_move(board, depth):
    loop = asyncio.get_event_loop()
    move = await loop.run_in_executor(None, select_move, board.copy(), depth)
    return move

# ==========================
# Нейронка
# ==========================

# import tensorflow as tf
# # model = tf.keras.models.load_model('chess_model.h5')
#
# def predict_move(board, model):
#     input_tensor = board_to_tensor(board)
#     input_tensor = np.expand_dims(input_tensor, axis=0)  # батч
#     logits = model.predict(input_tensor)
#     move_idx = np.argmax(logits[0])
#     # преобразуйте move_idx обратно в UCI или ход
#     move = index_to_move(move_idx)

# ==========================
# 6. Игра против движка
# ==========================

 # Реализация хода игроком
def get_square_under_mouse(pos):
    square_size = 80  # размер клетки в пикселях
    x, y = pos
    col = x // square_size
    row = y // square_size
    if 0 <= col < 8 and 0 <= row < 8:
        # Переводим в индекс
        square = (7 - row) * 8 + col
        return square
    return None

def highlight_square(screen, square, color=(0,255,0)):
    square_size = 80
    col = square % 8
    row = 7 - (square // 8)
    rect = pygame.Rect(col * square_size, row * square_size, square_size, square_size)
    pygame.draw.rect(screen, color, rect, 3)  # рамка толщиной 3 пикселя


async def play_game(depth=4):
    # Инициализация
    SQUARE_SIZE = 80
    pygame.init()
    screen = pygame.display.set_mode((SQUARE_SIZE*8, SQUARE_SIZE*8))
    pygame.display.set_caption("Chess")

    # Настройки сторон
    white_player = 'bot'  # или 'bot'
    black_player = 'bot'    # или 'human'
    board = chess.Board()

    selected_square = None
    valid_moves_for_selected = []


    def draw_board():
        colors = ["#f0d9b5", "#b58863"]
        for r in range(8):
            for c in range(8):
                color = colors[(r + c) % 2]
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(screen, color, rect)

                square = chess.square(c, 7 - r)
                piece = board.piece_at(square)
                if piece:
                    img = PIECE_IMAGES.get(piece.symbol())
                    if img:
                        img_scaled = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                        screen.blit(img_scaled, (c * SQUARE_SIZE, r * SQUARE_SIZE))
        # Выделение выбранной клетки
        if selected_square is not None:
            c = selected_square % 8
            r = 7 - (selected_square // 8)
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), rect, 3)

        # Выделение допустимых ходов
        for move in valid_moves_for_selected:
            c = move.to_square % 8
            r = 7 - (move.to_square // 8)
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, (255, 255, 0), rect, 3)

        pygame.display.flip()


    async def process_bot_move():
        move = await async_select_move(board, depth)
        print(f"{'Белые' if board.turn == chess.WHITE else 'Черные'} делают ход: {board.san(move)}")
        board.push(move)
        draw_board()

    async def async_select_move(board, depth):
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(None, select_move, board.copy(), depth)
        return move

    running = True
    while running:
        draw_board()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Обработка клика
                if ((board.turn == chess.WHITE and white_player == 'human') or
                    (board.turn == chess.BLACK and black_player == 'human')):
                    mouse_pos = pygame.mouse.get_pos()
                    c = mouse_pos[0] // SQUARE_SIZE
                    r = mouse_pos[1] // SQUARE_SIZE
                    square_clicked = chess.square(c, 7 - r)

                    piece = board.piece_at(square_clicked)
                    if selected_square is None:
                        # выбираем фигуру
                        if piece and ((piece.color and board.turn == chess.WHITE) or
                                      (not piece.color and board.turn == chess.BLACK)):
                            selected_square = square_clicked
                            valid_moves_for_selected = [
                                move for move in board.legal_moves if move.from_square == square_clicked
                            ]
                    else:
                        # делаем ход
                        move = chess.Move(selected_square, square_clicked)
                        if move in valid_moves_for_selected:
                            board.push(move)
                        selected_square = None
                        valid_moves_for_selected = []

        # Проверка окончания игры
        if board.is_game_over():
            print("Игра окончена:", board.result())
            break

        # Если текущий ход за бота
        if not board.is_game_over() and (
                (board.turn == chess.WHITE and white_player == 'bot') or
                (board.turn == chess.BLACK and black_player == 'bot')
        ):
            await asyncio.sleep(0.5)  # короткая задержка для плавности
            await process_bot_move()

        await asyncio.sleep(0.01)  # небольшая задержка для снижения нагрузки


# =============================
# 7. Интеграция + запуск
# =============================

if __name__ == '__main__':
    # Загрузка партий
    # pgn_filename = 'lichess_db_standard_rated_2025-01.pgn'  # укажите путь к файлу
    # print("Загрузка партий...")
    # games = load_games(pgn_filename, limit=500)  # можно изменить лимит
    #
    # print("Извлечение позиций...")
    # positions = extract_positions(games, max_moves=60)

    # Тут можно подготовить данные для обучения нейросети, пропускаем для простоты

    # Запуск игры
    asyncio.run(play_game())
    # play_self_play()