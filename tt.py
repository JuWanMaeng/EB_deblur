N = 3
grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]

visited = [[False] * N for _ in range(N)]
ans = 0
dr = [0, 0, 1, -1]
dc = [1, -1, 0, 0]

def dfs(r, c):
    global ans
    print(f"🔵 Enter ({r},{c})")
    
    if (r, c) == (2, 2):
        ans += 1
        print(f"  ✅ Reached goal at ({r},{c}), total paths: {ans}")
        return

    visited[r][c] = True

    for i in range(4):
        nr = r + dr[i]
        nc = c + dc[i]

        if 0 <= nr < N and 0 <= nc < N:
            if grid[nr][nc] == 0 and not visited[nr][nc]:
                print(f"    → Move to ({nr},{nc}) from ({r},{c})")
                dfs(nr, nc)

    visited[r][c] = False
    print(f"🔙 Backtrack from ({r},{c}) — unvisit\n")

# 시작점에서 DFS 호출
dfs(0, 0)
print(f"\n총 경로 수: {ans}")
