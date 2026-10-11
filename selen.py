from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess.pgn
import numpy as np




class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(17,128,kernel_size= 3, padding= 1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128,128, kernel_size=3, padding= 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.dropout =nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 4096)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
def board_to_tensor(board):
        tensor = np.zeros((17,8,8), dtype=np.int8)


        # 0-11
        for i in chess.SQUARES:
            piece = board.piece_at(i)
            if piece:
                layer_indx = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    layer_indx += 6

                row = chess.square_rank(i)
                col = chess.square_file(i)
                tensor[layer_indx, row, col] = 1.0

        # 12

        if board.turn == chess.WHITE:
            tensor[12,:,:] = 1.0

        # 13-16

        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0

        return tensor

def get_index(piece, el_size):
    """ИЩЕТ ИНДЕКС В ЗАВИСИМОСТИ ОТ КОРДИНАТ ГДЕ НАХОДИТСЯ ФИГУРА, ЧТОБЫ С ЕГО ПОМОЩЬЮ ДОБАВИТЬ ФИГУРУ НА ВИРТУАЛЬНУЮ ДОСКУ"""
    style_string = piece.get_attribute("style")
    if not style_string:
        return None
    
    coords = re.findall(r'\d+\.\d+|\d+', style_string)
    x = round(float(coords[0])/ el_size)
    y = 7 - round(float(coords[1])/ el_size)
    return y * 8 + x

def load_pieces(board,driver, el_size):
    """НАХОДИТ ВСЕ ФИГУРЫ В БРАУЗЕРЕ И ДОБАВЛЯЕТ ИХ НА ВИРТУАЛЬНУЮ ДОСКУ"""
    # if not isinstance(el_size, float) or not isinstance(el_size, int):
    #     raise ValueError("must be int")
    
    piece_map = {
        'pawn': chess.PAWN,
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'queen': chess.QUEEN,
        'king': chess.KING
    }

    pieces = driver.find_elements(By.TAG_NAME, "piece")
    
    for el in pieces:
        classes = el.get_attribute("class")
        if "ghost" in classes or "anim" in classes or "dragging" in classes or "fading" in classes:
            continue
        color = chess.WHITE if "white" in classes else chess.BLACK
        p_type = None

        for name, type in piece_map.items():
            if name in classes:
                p_type = type

        
        if p_type:
            indx = get_index(el,el_size)
            if indx is not None:
                board.set_piece_at(indx, chess.Piece(p_type, color))

    


def selen_main(path, model):
    #       ИНИЦИАЛИЗАЦИЯ
    driver = webdriver.Chrome()
    # actions = ActionChains(driver)

    driver.get(path)
    time.sleep(25)
    last_fen = ""
    #        НЕОБХОДИМЫЕ ДАННЫЕ С БРАУЗЕРА
    for _ in range(10000):
        board = chess.Board(None)
        container = driver.find_element(By.TAG_NAME, "cg-container")
        width = container.size['width']
        el_size = width / 8 
        #        ОСНОВНОЙ КОД



        load_pieces(board,driver,el_size)
        current_fen = board.board_fen()

        if current_fen == last_fen:
            time.sleep(0.1)
            continue
        last_fen = current_fen # Запоминаем новую позицию
        print("\n=== НОВАЯ ПОЗИЦИЯ ===")

        feature = board_to_tensor(board)
        feature = torch.from_numpy(feature).float()
        feature = feature.unsqueeze(0)

        model.eval()
        out = model(feature)
        mask = torch.full((4096,), -float('inf')).to(out.device)

        # Разрешаем только те ходы, которые легальны в python-chess
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            mask[idx] = 0 # Обнуляем штраф для легального хода

        # Складываем логиты с маской и выбираем лучший
        masked_out = out + mask
        predicted_idx = torch.argmax(masked_out).item()
        
        # _, predicted = torch.max(out.data,1)

        print(board)

        index = predicted_idx

        from_square = index // 64  # Целочисленное деление
        to_square = index % 64    # Остаток от деления
        move = chess.Move(from_square, to_square)
        

        print(f"Из клетки: {from_square}") 
        print(f"В клетку: {to_square}")   
        print(f"ход: {move.uci()}")
        



        # king_locator = By.CSS_SELECTOR, "piece.white.king"
        # target_king = driver.find_element(*king_locator)
        # actions.move_to_element(target_king).click().perform()

        
    driver.quit()
    







if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet()
    model.load_state_dict(torch.load('./models/models_firstmodel.pth', map_location=device))
    
    selen_main("https://lichess.org/YfBuo0ti", model)