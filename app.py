import streamlit as st
import numpy as np
import pandas as pd


grid = np.array([[0 for _ in range(7)] for _ in range(7)])
whites = {1: (0,1), 2: (1,2), 3: (2,1), 4: (4,1), 5: (5,2), 6: (6,1)}
blacks = {1: (0,5), 2: (1,4), 3: (2,5), 4: (4,5), 5: (5,4), 6: (6,5)}
obtacles = [(3,0),(3,6)]
goal = (3,3)

directions = {"left": (0,-1), "right": (0,1), "up": (-1,0),"down": (1,0)}

def init_grid():
    global grid
    grid = np.array([[0 for _ in range(7)] for _ in range(7)])
    
    grid[goal] = 9
    for obtacle in obtacles:
        grid[obtacle] = 5
    
    for w,b in zip(whites.values(),blacks.values()):
        grid[w] = 1
        grid[b] = 2
    
    return grid

def move(side, piece, direction):
    grid = st.session_state.grid
    whites = st.session_state.whites
    blacks = st.session_state.blacks

    if side == "white":
        selected_coord = whites[piece]
    else:
        selected_coord = blacks[piece]
    grid[selected_coord] = 0
    y, x = selected_coord
    iy, ix = directions[direction]
    is_out = False
    while True:
        if 0 > y+iy or y+iy >= 7 or 0 > x+ix or x+ix >= 7:
            break
        if grid[(y+iy, x+ix)] in [0,9]:
            y += iy
            x += ix
        else:
            break
    if grid[(y, x)] == 9:
            is_out = True
            if side == "white":
                del whites[piece]
            else:
                del blacks[piece]
            
            # whites에서 해당 piece 지워야함함
            return is_out
    new_coord = (y,x)
    if side == "white":
        whites[piece] = new_coord
        grid[new_coord] = 1
    else:
        blacks[piece] = new_coord
        grid[new_coord] = 2
    print(grid)

emoji_map = {0: "🔲", 1: "🔵", 2: "⚫", 5: "🟫",9: "🟪"}

# def show(grid, container):
#     emoji_array = [[emoji_map[val] for val in row] for row in grid]
#     container.empty()
#     with container:
#         for row in emoji_array:
#             st.write("".join(row))

def show(grid, container):
    emoji_array = ["".join([emoji_map[val] for val in row]) for row in grid]  # 각 행을 문자열로 변환
    container.empty()  # 기존 내용을 지움
    with container:
        st.text("\n".join(emoji_array))  # 전체 맵을 한 번에 출력


if "grid" not in st.session_state:
    st.session_state.grid = init_grid()
if "whites" not in st.session_state:
    st.session_state.whites = whites.copy()
if "blacks" not in st.session_state:
    st.session_state.blacks = blacks.copy()
if "side" not in st.session_state:
    st.session_state.side = "white"  # Default side



map_container = st.empty()
# map_container = st.container()  # 맵을 고정할 컨테이너
show(st.session_state.grid, map_container)

side = st.radio("말의 색상을 선택하세요:", options=["white", "black"], index=0)
st.session_state.side = side  # 상태 유지

st.write(side)

# 말 선택
default_piece = 1  # 기본값 설정

if side == "white":
    piece = st.selectbox(
        "말을 선택하세요 (1-6):", 
        options=list(st.session_state.whites.keys())
    )
else:
    piece = st.selectbox(
        "말을 선택하세요 (1-6):", 
        options=list(st.session_state.blacks.keys())
    )


# 방향 선택
default_direction = "right"  # 기본값 설정
direction = st.selectbox("이동할 방향을 선택하세요:", options=list(directions.keys()), index=list(directions.keys()).index(default_direction))


if st.button("move"):
    move(side, piece, direction)
    show(st.session_state.grid, map_container)

if st.button("comp"):
    pass


# 모델 예측 및 행동 수행
if st.button("Model"):
    # 현재 상태의 맵 입력 (flattened 상태로 모델 입력)
    state = st.session_state.grid.flatten()
    side = "black"

    # 모델 추론 (가정: predict 함수가 모델 행동을 예측)
    from model import predict, RLModel  # 모델 가져오기
    # 모델 초기화
    model = RLModel(input_dim=49, action_dim=24)  # input_dim: 상태 크기, action_dim: 최대 행동 크기 (6개 말 × 4 방향)

    # 가중치 로드 (학습된 모델 파일 경로)
    model.load_state_dict(torch.load("models/dqn_model.pth"))

    # 현재 상태와 유효한 말 설정
    valid_pieces = list(st.session_state.blacks.keys())

    # 추론
    piece, direction = predict(model, state, valid_pieces)
    print(f"모델 예측 결과: 말 {piece}, 방향 {direction}")

    # 이동 수행
    move("black", piece, direction)
    show(st.session_state.grid, map_container)