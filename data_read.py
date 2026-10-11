
import chess.pgn
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def get_random_game(filename):
    file_size = os.path.getsize(filename)

    with open(filename, "r", encoding="utf-8") as f:
        # Прыгаем на случайный байт в файле
        random_offset = random.randint(0, file_size - 1000)
        f.seek(random_offset)

        # Мы могли попасть в середину текста.
        # Нужно найти начало следующей игры (заголовок [Event)
        while True:
            line = f.readline()
            if not line: # Конец файла
                f.seek(0)
                break
            if line.startswith("[Event "):
                # Мы нашли начало игры. Теперь нужно переместить указатель
                # в начало этой строки, чтобы read_game ее считал.
                f.seek(f.tell() - len(line.encode('utf-8')) - 1)
                break

        return chess.pgn.read_game(f)


class ChessDataset(Dataset):
    def __init__(self, pgn_file):
        # self.f = pgn_file
        # self.offsets = []

        # with open(pgn_file, 'r') as f:
        #     while True:
        #         offset = f.tell()
        #         line = f.readline()
        #         if not line:
        #             break
        #         if line.startswith('[Event'):
        #             self.offsets.append(offset)

        self.features = np.load(os.path.join(pgn_file, "features.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(pgn_file, "labels.npy"), mmap_mode='r')
        self.results = np.load(os.path.join(pgn_file, 'results.npy'), mmap_mode='r')


    def __len__(self):
        # return len(self.offsets)
        return len(self.labels)

    def __getitem__(self, index):
        feature_np = self.features[index].copy() 
        
        feature = torch.from_numpy(feature_np).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        results = torch.tensor(self.results[index], dtype=torch.float32)
        return feature, label, results
    #     offset = self.offsets[index]

    #     with open(self.f, 'r') as f:
    #         f.seek(offset)
    #         game = chess.pgn.read_game(f)


    #     moves = list(game.mainline_moves())
    #     r_move_indx = random.randint(0, len(moves) - 1)

    #     target_move = moves[r_move_indx]

    #     board = game.board()
    #     for i, move in enumerate(moves):
    #         if i == r_move_indx:
    #             break
    #         board.push(move)


    #     features = self.board_to_tensor(board)

    #     # res_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    #     # result = res_map.get(game.headers.get("Result"), "1/2-1/2")

    #     # return features, torch.tensor(result, dtype= torch.float32)


    #     move_idx = target_move.from_square * 64 + target_move.to_square
    #     return features, torch.tensor(move_idx, dtype=torch.long)


    # def board_to_tensor(self,board):
    #     tensor = np.zeros((17,8,8), dtype=np.float32)


    #     # 0-11
    #     for i in chess.SQUARES:
    #         piece = board.piece_at(i)
    #         if piece:
    #             layer_indx = piece.piece_type - 1
    #             if piece.color == chess.BLACK:
    #                 layer_indx += 6

    #             row = chess.square_rank(i)
    #             col = chess.square_file(i)
    #             tensor[layer_indx, row, col] = 1.0

    #     # 12

    #     if board.turn == chess.WHITE:
    #         tensor[12,:,:] = 1.0

    #     # 13-16

    #     if board.has_kingside_castling_rights(chess.WHITE):
    #         tensor[13, :, :] = 1.0
    #     if board.has_queenside_castling_rights(chess.WHITE):
    #         tensor[14, :, :] = 1.0
    #     if board.has_kingside_castling_rights(chess.BLACK):
    #         tensor[15, :, :] = 1.0
    #     if board.has_queenside_castling_rights(chess.BLACK):
    #         tensor[16, :, :] = 1.0

    #     return torch.from_numpy(tensor)


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(17,128,kernel_size= 3, padding= 1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128,128, kernel_size=3, padding= 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()


        # firsst head (POLICY)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.dropout =nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 4096)

        #second head (VALUE)
        self.fc_value1 = nn.Linear(128 * 8 * 8, 512)
        self.fc_value2 = nn.Linear(512, 1)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)

        policy = F.relu(self.fc1(x))
        policy = self.dropout(policy)
        policy = self.fc2(policy)

        value = F.relu(self.fc_value1(x))
        value = self.dropout(value)
        value = F.tanh(self.fc_value2(value))
        return policy, value



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"обучение на: {device}")
#datas = ChessDataset("chessdata/lichess_elite_2020-01.pgn")
datas = ChessDataset("processed_data")


train_size = int(0.9 * len(datas))
val_size = len(datas) - train_size

# 3. Случайным образом делим данные
train_dataset, val_dataset = random_split(datas, [train_size, val_size])

# 4. Создаем лоадеры
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8)



model = ChessNet().to(device)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr= 0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

print("START TRAIN")
epochs = 100
best_avg_loss_val = 10.0
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()
    all_preds = []
    all_labels = []
    running_loss = 0

    for features, move_idx, results in train_loader:
        features, move_idx = features.to(device), move_idx.to(device), results.to(device)
        optimizer.zero_grad()

        policy,val = model(features)
        loss_policy = criterion(policy, move_idx)
        loss_val = criterion2(val.squeeze(), results)

        total_loss = loss_val + loss_policy
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        _, predicted = torch.max(out.data,1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(move_idx.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    acc = accuracy_score(all_preds, all_labels) * 100
    print(f"[{epoch}] - acc: {acc:.2f}% avg_loss: {avg_train_loss:.2f}%")

    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0
    with torch.no_grad():
        for features, move_idx in val_loader:
            features, move_idx = features.to(device), move_idx.to(device)
            out = model(features)
            loss = criterion(out, move_idx)

            running_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(move_idx.cpu().numpy())

        
        avg_loss = running_loss / len(val_loader)
        scheduler.step(avg_loss)
        acc = accuracy_score(all_labels, all_preds) * 100

        print(f"[VAL] - acc: {acc:.2f}% avg_loss: {avg_train_loss:.2f}%")

        if best_avg_loss_val > avg_loss:
            best_avg_loss_val = avg_loss
            torch.save(model.state_dict(), 'models/firstmodel.pth')
            print("Новая ЛУЧШАЯ модель! Сохраняем...")
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1

        if epochs_without_improvement > 5:
            print("EARLY STOP!")
            break

        print(f"---------------------------------------------")






