import chess
import chess.pgn
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

# 1. Загрузка и предобработка данных

def load_data_from_pgn(pgn_file, num_games=None):
    """
    Загружает шахматные партии из PGN-файла и преобразует их в данные, пригодные для обучения нейронной сети.

    Args:
        pgn_file (str): Путь к PGN-файлу.
        num_games (int, optional): Максимальное количество партий для загрузки. Если None, загружаются все партии. Defaults to None.

    Returns:
        tuple: Два списка:
            - positions (list): Список позиций (представленных в виде числовых массивов).
            - moves (list): Список соответствующих ходов (представленных в виде one-hot encoding).
    """
    positions = []
    moves = []
    game_count = 0

    pgn = open(pgn_file)

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        board = game.board()
        for move in game.mainline_moves():
            positions.append(board.fen())  # Используем FEN для представления позиции
            moves.append(move.uci())  # Используем UCI для представления хода
            board.push(move)

        game_count += 1
        if num_games is not None and game_count >= num_games:
            break

    print(f"Загружено {game_count} партий из {pgn_file}")
    return positions, moves

def fen_to_array(fen):
    """
    Преобразует FEN-представление шахматной позиции в числовой массив.
    """
    piece_map = {
        'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'r': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'n': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'b': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'N': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'B': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    board_array = []
    for row in fen.split(' ')[0].split('/'):
        for cell in row:
            if cell in piece_map:
                board_array.extend(piece_map[cell])
            elif cell.isdigit():
                for _ in range(int(cell)):
                    board_array.extend([0] * 12)  # Пустая клетка
    return np.array(board_array)


def move_to_one_hot(move_uci):
    """Преобразует ход в формате UCI в one-hot encoding."""
    # Создайте словарь для отображения UCI ходов в индексы
    all_possible_moves = []
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            all_possible_moves.append(chess.Move(from_square, to_square).uci())
            # Исправлено: используем chess.QUEEN, chess.ROOK и т.д.
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                all_possible_moves.append(chess.Move(from_square, to_square, promotion=promotion).uci())

    move_to_index = {move: i for i, move in enumerate(all_possible_moves)}

    one_hot = np.zeros(len(move_to_index))
    if move_uci in move_to_index:
        one_hot[move_to_index[move_uci]] = 1
    return one_hot

def prepare_data(positions, moves):
    """
    Преобразует позиции и ходы в формат, пригодный для обучения нейронной сети.
    """
    X = np.array([fen_to_array(fen) for fen in positions])
    y = np.array([move_to_one_hot(move) for move in moves])
    return X, y

# 2. Определение архитектуры нейронной сети
def create_model(input_shape, num_moves):
    """
    Создает модель нейронной сети.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_moves, activation='softmax')  # softmax для выбора хода
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Обучение нейронной сети
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Обучает модель нейронной сети.
    """
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val))
    return history

# 4. Сохранение обученной модели
def save_model(model, filepath):
    """
    Сохраняет обученную модель.
    """
    model.save(filepath)
    print(f"Модель сохранена в {filepath}")

# Основной код

if __name__ == "__main__":
    pgn_file = "lichess_db_standard_rated_2025-01.pgn"  # Замените на путь к вашему PGN-файлу
    model_filepath = "chess_model.keras"

    # Загрузка данных (ограничьте количество игр для начала)
    positions, moves = load_data_from_pgn(pgn_file, num_games=1000)

    # Подготовка данных
    X, y = prepare_data(positions, moves)

    # Разделение на обучающую и валидационную выборки
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Определение архитектуры модели
    input_shape = X_train.shape[1]
    num_moves = y_train.shape[1]
    model = create_model(input_shape, num_moves)

    # Обучение модели
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=2, batch_size=64) # Уменьшите epochs для быстрого теста

    # Сохранение модели
    save_model(model, model_filepath)