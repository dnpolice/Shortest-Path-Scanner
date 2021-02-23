
def breadthFirstSearch(matrix, start, end):
    startRowIdx = start[0]
    startColIdx = start[1]
    endRowIdx = end[0]
    endColIdx = end[1]
    #[wall==1, row, col, visited==1, prev_row, prev_col]
    board = [[[matrix[row][col], row, col, 0, -1, -1] for col in range(0, len(matrix[0]))] for row in range(0, len(matrix))]
    start_node = board[startRowIdx][startColIdx]
    start_node[3] = 1
    queue = [start_node]
    success = False
    end_node = None
    while len(queue) > 0:
        node = queue[0]
        queue.pop(0)
        if endRowIdx == node[1] and endColIdx == node[2]:
            success = True
            end_node = node
            break

        if (node[2]-1) >= 0 and board[node[1]][node[2]-1][0] != 1 and board[node[1]][node[2]-1][3] != 1:
            next_node = board[node[1]][node[2]-1]
            next_node[4] = node[1]
            next_node[5] = node[2]
            next_node[3] = 1
            queue.append(next_node)
        if (node[2] + 1) < len(board[0]) and board[node[1]][node[2] + 1][0] != 1 and board[node[1]][node[2]+1][3] != 1:
            next_node = board[node[1]][node[2]+1]
            next_node[4] = node[1]
            next_node[5] = node[2]
            next_node[3] = 1
            queue.append(next_node)
        if (node[1] - 1) >= 0 and board[node[1]-1][node[2]][0] != 1 and board[node[1]-1][node[2]][3] != 1:
            next_node = board[node[1]-1][node[2]]
            next_node[4] = node[1]
            next_node[5] = node[2]
            next_node[3] = 1
            queue.append(next_node)
        if (node[1] + 1) < len(board[0]) and board[node[1]+1][node[2]][0] != 1 and board[node[1]+1][node[2]][3] != 1:
            next_node = board[node[1]+1][node[2]]
            next_node[4] = node[1]
            next_node[5] = node[2]
            next_node[3] = 1
            queue.append(next_node)
    if success:
        return create_path(board, end_node)
    return None


def create_path(board, node):
    # [wall==1, row, col, visited==1, prev_row, prev_col]
    # [[row,col]...]
    path_array = []
    while node[4] != -1:
        path_array.insert(0, [node[1], node[2]])
        node = board[node[4]][node[5]]
    path_array.insert(0, [node[1], node[2]])
    return path_array
