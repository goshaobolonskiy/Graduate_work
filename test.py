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

def draw_board(board):
    # Рисуем фон
    colors = [("#f0d9b5", "#b58863")][0]
    for r in range(8):
        for c in range(8):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
            # Надписи уже не нужны (используем изображения)
            square = chess.square(c, 7 - r)
            piece = board.piece_at(square)
            if piece:
                img = PIECE_IMAGES[piece.symbol()]
                img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(img, (c * SQUARE_SIZE, r * SQUARE_SIZE))
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

# =======================
# 4. Минимакс с альфа-бета
# =======================

def minimax(board, depth, alpha, beta, is_maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    if is_maximizing:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# ==========================
# 5. Выбор лучшего хода
# ==========================

def select_move(board, depth):
    best_move = None
    best_score = -np.inf
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# ==========================
# 6. Игра против движка
# ==========================

def user_move(board):
    while True:
        move = input("Ваш ход (например, e2e4): ")
        try:
            move_obj = board.parse_san(move)
        except ValueError:
            try:
                move_obj = board.parse_uci(move)
            except ValueError:
                print("Некорректный ход, попробуйте снова.")
                continue
        if move_obj in board.legal_moves:
            return move_obj
        else:
            print("Недопустимый ход. Попробуйте снова.")


def play_game():
    board = chess.Board()
    print("Начинаем игру! Введите 'exit' для выхода.")
    while not board.is_game_over():
        print("\nТекущая позиция:\n", board)
        draw_board(board)  # добавляем визуализацию

        # Остальной код...
        human_move_input = input("Ваш ход (или 'auto' для движка): ")
        if human_move_input == 'exit':
            break
        elif human_move_input == 'auto':
            move = select_move(board, depth=3)
            print(f"Движок совершает ход: {board.san(move)}")
            board.push(move)
        else:
            try:
                move = board.parse_san(human_move_input)
            except:
                try:
                    move = board.parse_uci(human_move_input)
                except:
                    print("Некорректный ход, попробуйте ещё раз.")
                    continue
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Недопустимый ход.")
                continue

        if board.is_game_over():
            break

        # Ход движка
        print("Ход движка...")
        move = select_move(board, depth=3)
        print(f"Движок делает ход: {board.san(move)}")
        board.push(move)

    print("Игра окончена:", board.result())


def play_self_play():
    board = chess.Board()
    auto_play = True  # Можно сделать настраиваемым
    depth = 4
    running = True
    clock = pygame.time.Clock()

    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # Обновление доски
        draw_board(board)

        if not board.is_game_over():
            if auto_play:
                move = select_move(board, depth=depth)
                print(f"Движок делает ход: {board.san(move)}")
                board.push(move)
            else:
                # В ручном режиме — тут можно добавить обработку ввода
                pass
        else:
            break

        # Контроль скорости обновлений
        clock.tick(1)  # обновляем 1 раз в секунду или по желанию

    print("Игра окончена:", board.result())

# =============================
# 7. Интеграция + запуск
# =============================

if __name__ == '__main__':
    # Загрузка партий
    pgn_filename = 'lichess_db_standard_rated_2025-01.pgn'  # укажите путь к файлу
    print("Загрузка партий...")
    games = load_games(pgn_filename, limit=500)  # можно изменить лимит

    print("Извлечение позиций...")
    positions = extract_positions(games, max_moves=60)

    # Тут можно подготовить данные для обучения нейросети, пропускаем для простоты

    # Запуск игры
    # play_game()
    play_self_play()