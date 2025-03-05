import chess
import pygame
import sys
import numpy as np
from tensorflow import keras
import time

# Загрузка модели
model = keras.models.load_model('chess_model.keras')

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
    if board.piece_type_at(move.from_square) == chess.PAWN:
        if (board.color_at(move.from_square) == chess.WHITE and move.to_square // 8 == 7) or \
           (board.color_at(move.from_square) == chess.BLACK and move.to_square // 8 == 0):
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move

def fen_to_array(fen):
    """Преобразует FEN в numpy array."""
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
def predict_move(board, model):
    """Предсказывает лучший ход для данной позиции, используя модель."""
    fen = board.fen()
    input_data = fen_to_array(fen).reshape(1, -1)  # reshape для соответствия входной форме модели
    predictions = model.predict(input_data)[0]  # Получаем вероятности для всех ходов

    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]

    # Создаем список возможных ходов в формате UCI
    all_possible_moves = []
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            all_possible_moves.append(chess.Move(from_square, to_square).uci())
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                all_possible_moves.append(chess.Move(from_square, to_square, promotion=promotion).uci())

    # Находим индексы допустимых ходов в списке всех возможных ходов
    legal_move_indices = [i for i, move in enumerate(all_possible_moves) if move in legal_moves_uci]

    # Выбираем лучший ход из допустимых на основе предсказаний модели
    best_move_index = np.argmax(predictions[legal_move_indices])
    best_legal_move_index = legal_move_indices[best_move_index]

    return all_possible_moves[best_legal_move_index]

def display_game_over_message(message):
    font = pygame.font.Font(None, 74)
    text = font.render(message, True, (0, 0, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.wait(3000)  # Ждем 3 секунды перед выходом

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
                            board.push(move)  # Выполняем ход на доске
                            dragging_piece = None
                            selected_square = None

                            # Проверка на окончание игры
                            if board.is_checkmate():
                                draw_board(board)
                                display_game_over_message("Мат! Белые выиграли!")
                                pygame.quit()
                                sys.exit()
                            elif board.is_stalemate():
                                draw_board(board)
                                display_game_over_message("Ничья!")
                                pygame.quit()
                                sys.exit()
                            elif board.is_insufficient_material():
                                draw_board(board)
                                display_game_over_message("Ничья! Мало материала!")
                                pygame.quit()
                                sys.exit()

                            # Ход AI
                            try:
                                ai_move_uci = predict_move(board, model)
                                if ai_move_uci:
                                    ai_move = chess.Move.from_uci(ai_move_uci)
                                    board.push(ai_move)
                            except Exception as e:
                                print(f"Ошибка при ходе AI: {e}")
                                #Если AI выдает ошибку, можно завершить игру или сделать случайный ход
                                #Например:
                                #legal_moves = list(board.legal_moves)
                                #if legal_moves:
                                #    board.push(random.choice(legal_moves))
                                #else:
                                #    display_game_over_message("Ничья из-за отсутствия ходов!")
                                #    pygame.quit()
                                #    sys.exit()
                            # Проверка на окончание игры после хода AI
                            if board.is_checkmate():
                                draw_board(board)
                                display_game_over_message("Мат! Черные выиграли!")
                                pygame.quit()
                                sys.exit()
                            elif board.is_stalemate():
                                draw_board(board)
                                display_game_over_message("Ничья!")
                                pygame.quit()
                                sys.exit()
                            elif board.is_insufficient_material():
                                draw_board(board)
                                display_game_over_message("Ничья! Мало материала!")
                                pygame.quit()
                                sys.exit()
                        else:
                            dragging_piece = None
                            selected_square = None

            if event.type == pygame.MOUSEMOTION:
                if dragging_piece:  # Обработка перетаски
                    screen.fill((255, 255, 255))
                    draw_board(board)  # Отрисовываем доску
                    mouse_x, mouse_y = event.pos
                    piece_image = PIECE_IMAGES[dragging_piece.symbol()]
                    piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                    screen.blit(piece_image, (mouse_x - SQUARE_SIZE // 2, mouse_y - SQUARE_SIZE // 2))

            draw_board(board)
            pygame.display.flip()
            clock.tick(30)

if __name__ == '__main__':
    main()