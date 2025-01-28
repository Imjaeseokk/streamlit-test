import torch
import torch.nn as nn

class RLModel(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(RLModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Action을 말과 방향으로 변환
def decode_action(action, valid_pieces):
    """
    행동(action)을 유효한 말(valid_pieces)과 방향으로 변환합니다.
    """
    piece_idx = action // 4  # 유효한 말 중 선택된 인덱스
    if piece_idx >= len(valid_pieces):
        raise ValueError("Invalid action: Piece index out of range.")
    piece = valid_pieces[piece_idx]  # 유효한 말 번호
    direction_idx = action % 4  # 방향 인덱스
    directions = ["left", "right", "up", "down"]
    return piece, directions[direction_idx]


# 추론 함수
def predict(model, state, valid_pieces):
    """
    모델을 사용해 행동(action)을 예측하고,
    유효한 말(valid_pieces)과 방향을 반환합니다.

    Args:
        model: 학습된 RLModel.
        state: 현재 환경 상태 (Flatten된 상태).
        valid_pieces: 현재 유효한 말의 목록.
    Returns:
        piece: 선택된 말 번호.
        direction: 선택된 방향.
    """
    model.eval()
    with torch.no_grad():
        # 상태를 텐서로 변환
        logits = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

        # 행동(action) 선택
        action = torch.argmax(logits).item()  # 최대 확률의 행동 선택

        # 행동 디코딩 (유효한 말 기준)
        piece, direction = decode_action(action, valid_pieces)
        return piece, direction
