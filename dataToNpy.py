import numpy as np
import chess.pgn
import os

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

def preprocess_pgn(pgn_path, out_dir, max_games=40000):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_features = []
    all_labels = []
    all_results = []
    
    game_count = 0
    with open(pgn_path, "r") as pgn:
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            res_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
            result = res_map[game.headers.get("Result", "1/2-1/2")]



            board = game.board()
            for move in game.mainline_moves():
                # Сохраняем состояние до хода
                all_features.append(board_to_tensor(board))
                # Сохраняем ход
                all_labels.append(move.from_square * 64 + move.to_square)
                all_results.append(result)
                board.push(move)
            
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Обработано партий: {game_count}")

    # Сохраняем как массив NumPy
    np.save(os.path.join(out_dir, "features.npy"), np.array(all_features, dtype=np.int8))
    np.save(os.path.join(out_dir, "labels.npy"), np.array(all_labels, dtype=np.int64))
    np.save(os.path.join(out_dir, "results.npy"), np.array(all_results, dtype=np.float32))
    
    print(f"Готово! Сохранено {len(all_labels)} позиций.")


if __name__ == "__main__":
    preprocess_pgn("chessdata/lichess_elite_2020-01.pgn", "processed_data")
        # res_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
        # result = res_map.get(game.headers.get("Result"), "1/2-1/2")