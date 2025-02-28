import chess
import chess.pgn
import numpy as np
import tensorflow as tf


def board_to_array(board):
    """Преобразует шахматную доску в 3D массив для входа в нейросеть."""
    array = np.zeros((8, 8, 12))  # 12 слоев для фигур
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        layer = piece.piece_type - 1  # Индекс слоя соответствует типу фигуры
        color = 1 if piece.color == chess.WHITE else -1
        array[square // 8, square % 8, layer] = color  # Устанавливаем цвет фигуры

    return array


def generate_training_data_from_pgn(file_path):
    """Генерирует данные для обучения из файла PGN."""
    positions = []
    scores = []

    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            result = game.headers["Result"]

            for move in game.mainline_moves():
                positions.append(board_to_array(board))

                # Оценка результата партии
                if result == "1-0" and board.turn == chess.WHITE:
                    scores.append(1)  # Победа белых
                elif result == "0-1" and board.turn == chess.BLACK:
                    scores.append(1)  # Победа черных
                else:
                    scores.append(0)  # Поражение

                board.push(move)

    return np.array(positions), np.array(scores)


def create_model():
    """Создает модель нейронной сети."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')  # Оценка позиции
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def train_model(file_path):
    """Обучает модель на данных из файла PGN и сохраняет её."""
    model = create_model()
    positions, scores = generate_training_data_from_pgn(file_path)

    # Обучение модели с выводом подробной информации
    model.fit(positions, scores, epochs=10, batch_size=64, verbose=1)

    model.save('chess_ai_model.h5')  # Сохраняем модель


def main():
    pgn_file = 'test_games.pgn'  # Укажите путь к вашему PGN файлу
    train_model(pgn_file)


if __name__ == "__main__":
    main()