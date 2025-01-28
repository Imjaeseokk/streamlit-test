import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from tqdm import tqdm

# 환경 설정
BOARD_SIZE = 7  # 7x7 보드
ACTION_DIM = 6 * 4  # 말 개수 * 4 방향
STATE_DIM = BOARD_SIZE * BOARD_SIZE  # Flatten된 상태 크기
GAMMA = 0.99  # 할인율
LR = 0.001  # 학습률
BATCH_SIZE = 64  # 미니배치 크기
TARGET_UPDATE = 10  # 타겟 네트워크 업데이트 주기
MEMORY_CAPACITY = 10000  # 리플레이 메모리 크기

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 환경 초기화 함수
def reset_environment():
    global whites, blacks, grid
    whites = {1: (0, 1), 2: (1, 2), 3: (2, 1), 4: (4, 1), 5: (5, 2), 6: (6, 1)}
    blacks = {1: (0, 5), 2: (1, 4), 3: (2, 5), 4: (4, 5), 5: (5, 4), 6: (6, 5)}
    obstacles = [(3, 0), (3, 6)]
    goal = (3, 5)

    grid = np.zeros((BOARD_SIZE, BOARD_SIZE))
    grid[goal] = 9  # 목표
    for obtacle in obstacles:
        grid[obtacle] = 5
    for w, b in zip(whites.values(), blacks.values()):
        grid[w] = 1
        grid[b] = 2
    return grid.flatten()  # Flatten된 상태 반환

# 이동 함수
def move(piece, direction, side):
    global whites, blacks, grid
    directions = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}

    print("w:", whites)
    print("b:", blacks)

    if side == "white":
        selected_coord = whites[piece]
    else:
        selected_coord = blacks[piece]

    grid[selected_coord] = 0
    y, x = selected_coord
    dy, dx = directions[direction]

    while True:
        if 0 > y + dy or y + dy >= BOARD_SIZE or 0 > x + dx or x + dx >= BOARD_SIZE:
            break
        if grid[(y + dy, x + dx)] in [0, 9]:  # 이동 가능
            y += dy
            x += dx
        else:
            break

    new_coord = (y, x)
    if grid[new_coord] == 9:  # 목표 도달
        if side == "white":
            del whites[piece]
        else:
            del blacks[piece]
        return 1  # 성공 보상

    if side == "white":
        whites[piece] = new_coord
        grid[new_coord] = 1
    else:
        blacks[piece] = new_coord
        grid[new_coord] = 2

    return 0  # 일반 보상

# 행동 디코딩 함수
def decode_action(action, valid_pieces):
    piece_idx = action // 4  # 말의 인덱스 (0부터 시작)
    if piece_idx >= len(valid_pieces):
        raise ValueError("Invalid action: Piece index out of range.")
    piece = valid_pieces[piece_idx]  # 유효한 말 중 선택
    direction_idx = action % 4
    directions = ["left", "right", "up", "down"]
    return piece, directions[direction_idx]

# DQN 학습 루프
def train_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Q-Network 및 Target Network 초기화
    policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_net = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_CAPACITY)

    for episode in tqdm(range(1000), desc="Training Progress"):
        state = reset_environment()
        done = False
        total_reward = 0
        current_player = "white"  # 첫 번째 턴은 흰색 플레이어

        while not done:
            # Epsilon-Greedy 행동 선택
            epsilon = max(0.1, 0.9 - 0.01 * episode)
            if random.random() < epsilon:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax(dim=1).item()

            # 행동 수행 및 보상 계산
            valid_pieces = list(whites.keys() if current_player == "white" else blacks.keys())
            action = policy_net(state_tensor).argmax(dim=1).item()
            piece, direction = decode_action(action, valid_pieces)
            reward = move(piece, direction, current_player)  # 현재 플레이어가 행동
            total_reward += reward

            # 다음 상태 가져오기
            next_state = grid.flatten()
            done = len(whites) == 0 or len(blacks) == 0  # 모든 말이 골에 도달하면 종료

            # 경험 리플레이 저장
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # 플레이어 교대
            current_player = "black" if current_player == "white" else "white"

            # 학습
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1)

                # Q-값 계산
                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 타겟 네트워크 업데이트
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 모델 저장
    torch.save(policy_net.state_dict(), "models/dqn_model.pth")


if __name__ == "__main__":
    train_dqn()