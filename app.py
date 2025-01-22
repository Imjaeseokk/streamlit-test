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

def move(piece, direction):
    grid = st.session_state.grid
    whites = st.session_state.whites
    selected_coord = whites[piece]
    grid[selected_coord] = 0
    y, x = selected_coord
    iy, ix = directions[direction]
    is_out = False
    while True:
        if 0 > y+iy or y+iy >= 7 or 0 > x+ix or x+ix >= 7:
            break
        if grid[(y+iy, x+ix)] == 0:
            y += iy
            x += ix
        elif grid[(y+iy, x+ix)] == 9:
            is_out = True
            # whites에서 해당 piece 지워야함함
            return is_out
        else:
            break
    new_coord = (y,x)
    grid[new_coord] = 1
    whites[piece] = new_coord
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

map_container = st.empty()
# map_container = st.container()  # 맵을 고정할 컨테이너
show(st.session_state.grid, map_container)

# 말 선택
default_piece = 1  # 기본값 설정
piece = st.selectbox("말을 선택하세요 (1-6):", options=list(whites.keys()), index=list(whites.keys()).index(default_piece))

# 방향 선택
default_direction = "right"  # 기본값 설정
direction = st.selectbox("이동할 방향을 선택하세요:", options=list(directions.keys()), index=list(directions.keys()).index(default_direction))


if st.button("move"):
    move(piece, direction)
    show(st.session_state.grid, map_container)

if st.button("comp"):
    pass
