import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def obstacle_map(xStart, yStart, zStart, xTarget, yTarget, zTarget, MAX_X, MAX_Y, MAX_Z):
    """
    This function returns a map containing a random distribution of obstacles.
    
    Parameters:
    xStart (int): Starting x-coordinate.
    yStart (int): Starting y-coordinate.
    xTarget (int): Target x-coordinate.
    yTarget (int): Target y-coordinate.
    MAX_X (int): Maximum x-dimension of the map.
    MAX_Y (int): Maximum y-dimension of the map.
    
    Returns:
    np.ndarray: Array containing the coordinates of the starting point, obstacles, and the target point.
    """
    rand_map = np.random.rand(MAX_X, MAX_Y, MAX_Z)
    map = []
    map.append([xStart, yStart, zStart])  # Starting point

    obstacle_ratio = 0.05
    for i in range(MAX_X):
        for j in range(MAX_Y):
            for k in range(MAX_Z):
                if (rand_map[i, j, k] < obstacle_ratio and 
                    (i != xStart or j != yStart or k != zStart) and 
                    (i != xTarget or j != yTarget or k != zTarget)):
                    map.append([i, j, k])  # Add obstacle

    map.append([xTarget, yTarget, zTarget])  # Target point

    return np.array(map)  # Convert to NumPy array for easier handling

def visualize_map(map, path, container):
    """
    This function visualizes the 2D grid map consisting of obstacles, 
    start point, target point, and the optimal path.
    
    Parameters:
    map (np.ndarray): Coordinates of obstacles, start point, and target point.
    path (np.ndarray): The optimal path.
    visit_nodes (np.ndarray): Nodes visited during the pathfinding process (open/closed list).
    """
    plt.figure(figsize=(7, 7))
    
    sz_map = np.max(map) + 1
    
    # print(sz_map)
    # Obstacles
    obst_sz = max(2500 / sz_map, 36)
    obst_cnt = range(1, len(map) - 1)
    obst_color = np.array([55, 184, 157]) / 255
    plt.scatter(map[obst_cnt, 0] + 0.5, map[obst_cnt, 1] + 0.5, s=obst_sz, c=[obst_color], label='Obstacles', marker='o')

    # Start point
    plt.scatter(map[0, 0] + 0.5, map[0, 1] + 0.5, color='b', marker='*', label='Start Point')

    # Target point
    plt.scatter(map[-1, 0] + 0.5, map[-1, 1] + 0.5, color='r', marker='*', label='Target Point')

    # Optimal path
    if len(path) > 1:
        path_cnt = range(1, len(path) - 1)
        plt.scatter(path[path_cnt, 0] + 0.5, path[path_cnt, 1] + 0.5, color='b', label='Path', marker='o')
        plt.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, 'b', label='Optimal Path')

    if len(container.closed_list) > 1:
        # Open list (visited nodes, node state 1)
        node_sz = 8000 / sz_map
        # open_nodes = visit_nodes[visit_nodes[:, 0] == 1, 1:3]
        # scatter1 = plt.scatter(open_nodes[:, 0] - 0.5, open_nodes[:, 1] - 0.5, s=node_sz, color='g', marker='s', alpha=0.1, label='Open List')

        # Closed list (visited nodes, node state 0)
        closed_nodes = np.array(container.closed_list)
        scatter2 = plt.scatter(closed_nodes[:, 1] + 0.5, closed_nodes[:, 2] + 0.5, s=node_sz, color='b', marker='s', alpha=0.2, label='Closed List')

    plt.grid(True)
    plt.xticks(np.arange(0, sz_map+1, 1))
    plt.yticks(np.arange(0, sz_map+1, 1))
    plt.legend()
    plt.show()

def visualize_map_3d(map, path, container):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    
    sz_map = np.max(map) + 1
    
    # Obstacles
    obst_sz = max(2500 / sz_map, 36)
    obst_cnt = range(1, len(map) - 1)
    obst_color = np.array([55, 184, 157]) / 255
    ax.scatter(map[obst_cnt, 0] + 0.5, map[obst_cnt, 1] + 0.5, map[obst_cnt, 2] + 0.5, s=obst_sz, c=[obst_color], label='Obstacles', marker='o')

    # Start point
    ax.scatter(map[0, 0] + 0.5, map[0, 1] + 0.5, map[0, 2] + 0.5, color='b', marker='*', label='Start Point')

    # Target point
    ax.scatter(map[-1, 0] + 0.5, map[-1, 1] + 0.5, map[-1, 2] + 0.5, color='r', marker='*', label='Target Point')

    # Optimal path
    if len(path) > 1:
        path_cnt = range(1, len(path) - 1)
        ax.scatter(path[path_cnt, 0] + 0.5, path[path_cnt, 1] + 0.5, path[path_cnt, 2] + 0.5, color='b', label='Path', marker='o')
        ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, path[:, 2] + 0.5, 'b', label='Optimal Path')

    # if len(container.closed_list) > 1:
    #     # Open list (visited nodes, node state 1)
    #     node_sz = 8000 / sz_map
    #     # open_nodes = visit_nodes[visit_nodes[:, 0] == 1, 1:3]
    #     # scatter1 = plt.scatter(open_nodes[:, 0] - 0.5, open_nodes[:, 1] - 0.5, s=node_sz, color='g', marker='s', alpha=0.1, label='Open List')

    #     # Closed list (visited nodes, node state 0)
    #     closed_nodes = np.array(container.closed_list)
    #     ax.scatter(closed_nodes[:, 1] + 0.5, closed_nodes[:, 2] + 0.5, closed_nodes[:, 3] + 0.5, s=node_sz, color='b', marker='s', alpha=0.2, label='Closed List')

    plt.grid(True)
    plt.xticks(np.arange(0, sz_map+1, 1))
    plt.yticks(np.arange(0, sz_map+1, 1))
    plt.legend()
    plt.show()


def distance(x1, y1, z1, x2, y2, z2):
    """计算启发式距离（曼哈顿距离）"""
    # return abs(x1 - x2) + abs(y1 - y2)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

class Container:
    def __init__(self):
        self.open_list = []
        self.closed_list = []
    def insert_open(self, x, y, z, parent_x, parent_y, parent_z, h, g, f):
        """插入到OPEN列表中的节点"""
        self.open_list.append([1, x, y, z, parent_x, parent_y, parent_z, h, g, f]) # 格式: [is_on_list, x, y, z, parent_x, parent_y, parent_z, h, g, f]
    def insert_closed(self, x, y, z):
        """插入到CLOSED列表中的节点"""
        self.closed_list.append([0, x, y, z]) # 格式: [is_on_list, x, y, z]

    def min_f(self):
        """从OPEN列表中选择f值最小的节点"""
        min_f = 999999
        min_index = 0
        for i in range(len(self.open_list)):
            if self.open_list[i][7] < min_f and self.open_list[i][0] == 1:
                min_f = self.open_list[i][7]
                min_index = i
        return min_index

def is_in(x, y, z, list):
    """判断节点是否在OPEN或CLOSED列表中"""
    for i in range(len(list)):
        if list[i][1] == x and list[i][2] == y and list[i][3] == z:
            return True, i
    return False, -1


def expand_node(current_node, xTarget, yTarget, zTarget, MAP, container):
    px = current_node[1]
    py = current_node[2]
    pz = current_node[3]
    pg = current_node[8]
    expand_node_list = []
    expand_node_list.append([px - 1, py - 1, pz - 1])
    expand_node_list.append([px - 1, py - 1, pz])
    expand_node_list.append([px - 1, py - 1, pz + 1])
    expand_node_list.append([px - 1, py, pz - 1])
    expand_node_list.append([px - 1, py, pz])
    expand_node_list.append([px - 1, py, pz + 1])
    expand_node_list.append([px - 1, py + 1, pz - 1])
    expand_node_list.append([px - 1, py + 1, pz])
    expand_node_list.append([px - 1, py + 1, pz + 1])
    expand_node_list.append([px, py - 1, pz - 1])
    expand_node_list.append([px, py - 1, pz])
    expand_node_list.append([px, py - 1, pz + 1])
    expand_node_list.append([px, py, pz - 1])
    expand_node_list.append([px, py, pz + 1])
    expand_node_list.append([px, py + 1, pz - 1])
    expand_node_list.append([px, py + 1, pz])
    expand_node_list.append([px, py + 1, pz + 1])
    expand_node_list.append([px + 1, py - 1, pz - 1])
    expand_node_list.append([px + 1, py - 1, pz])
    expand_node_list.append([px + 1, py - 1, pz + 1])
    expand_node_list.append([px + 1, py, pz - 1])
    expand_node_list.append([px + 1, py, pz])
    expand_node_list.append([px + 1, py, pz + 1])
    expand_node_list.append([px + 1, py + 1, pz - 1])
    expand_node_list.append([px + 1, py + 1, pz])
    expand_node_list.append([px + 1, py + 1, pz + 1])

    for i, node in enumerate(expand_node_list):
        x = node[0]
        y = node[1]
        z = node[2]
        in_open, index_open = is_in(x, y, z, container.open_list)
        in_closed, index_closed = is_in(x, y, z, container.closed_list)
        
        h = distance(x, y, z, xTarget, yTarget, zTarget)
        g = pg
        # if i == 0 or i == 2 or i == 6 or i == 8 or i == 17 or i == 19 or i == 23 or i == 25:
        #     g += 1.7
        # elif i == 1 or i == 3 or i == 5 or i == 7 or i == 9 or i == 11 or i == 14 or i == 16 or i == 18 or i == 20 or i == 22 or i == 24:
        #     g += 1.4
        # else:
        #     g += 1
        g += distance(px, py, pz, x, y, z)
        f = h + g

        if x >= 0 and x < MAP.shape[0] and y >= 0 and y < MAP.shape[1] and z >= 0 and z < MAP.shape[2]:
            if MAP[x, y, z] != -1 and not in_closed:
                if not in_open:
                    container.insert_open(x, y, z, px, py, pz, h, g, f)
                else:
                    if f < container.open_list[index_open][9]:
                        container.open_list[index_open][4] = px
                        container.open_list[index_open][5] = py
                        container.open_list[index_open][6] = pz
                        container.open_list[index_open][7] = h
                        container.open_list[index_open][8] = g
                        container.open_list[index_open][9] = f


def initialize_map(map, MAX_X, MAX_Y, MAX_Z):
    """
    初始化地图、障碍物、起点和目标点。

    参数:
    map (np.ndarray): 含障碍物、起点、目标点的地图信息。
    MAX_X (int): 网格的x维度。
    MAX_Y (int): 网格的y维度。

    返回:
    MAP (np.ndarray): 更新后的2D网格地图。
    OPEN (list): 初始化后的OPEN列表。
    CLOSED (list): 初始化后的CLOSED列表。
    xStart, yStart, xTarget, yTarget (int): 起点和目标点的坐标。
    """

    size_map = map.shape[0]
    
    # 定义2D网格地图，障碍物=-1，目标点=0，起点=1
    MAP = 2 * np.ones((MAX_X, MAX_Y, MAX_Z))  # 初始化为2

    # 初始化目标点位置
    xTarget = int(map[size_map - 1, 0])
    yTarget = int(map[size_map - 1, 1])
    zTarget = int(map[size_map - 1, 2])
    MAP[xTarget, yTarget, zTarget] = 0
    
    # 初始化障碍物位置
    for i in range(1, size_map - 1):
        xval = int(map[i, 0])
        yval = int(map[i, 1])
        zval = int(map[i, 2])
        MAP[xval, yval, zval] = -1
    
    # 初始化起点位置
    xStart = int(map[0, 0])
    yStart = int(map[0, 1])
    zStart = int(map[0, 2])
    MAP[xStart, yStart, zStart] = 1
    return MAP, xStart, yStart, zStart, xTarget, yTarget, zTarget

def backtrace(current_node, container):
    path = []
    while current_node[4] != current_node[1] or current_node[5] != current_node[2] or current_node[6] != current_node[3]:
        path.append([current_node[1], current_node[2], current_node[3]])
        _, index = is_in(current_node[4], current_node[5], current_node[6], container.open_list)
        current_node = container.open_list[index]
    path.append([current_node[1], current_node[2], current_node[3]])
    return path



def A_star_search(map, MAX_X, MAX_Y, MAX_Z):
    MAP, xStart, yStart, zStart, xTarget, yTarget, zTarget= initialize_map(map, MAX_X, MAX_Y, MAX_Z)
    container = Container()
    # 初始化OPEN和CLOSED列表
    # 将所有障碍物添加到CLOSED列表
    for i in range(MAX_X):
        for j in range(MAX_Y):
            for k in range(MAX_Z):
                if MAP[i, j, k] == -1:
                    container.insert_closed(i, j, k)
    # 将起点作为第一个节点放入OPEN列表
    goal_distance = distance(xStart, yStart, zStart, xTarget, yTarget, zTarget)
    container.insert_open(xStart, yStart, zStart, xStart, yStart, zStart, goal_distance, 0, goal_distance)
    
    # # 将起点加入CLOSED列表
    # CLOSED.append([xStart, yStart])

    while(1):
        # 从OPEN列表中选择f值最小的节点
        min_index = container.min_f()
        current_node = container.open_list[min_index]
        container.open_list[min_index][0] = 0
        container.insert_closed(current_node[1], current_node[2], current_node[3])
        if current_node[1] == xTarget and current_node[2] == yTarget and current_node[3] == zTarget:
            print("Find the target!")
            return backtrace(current_node, container), container
        expand_node(current_node, xTarget, yTarget, zTarget, MAP, container)

        if len(container.closed_list) == MAX_X * MAX_Y * MAX_Z:
            print("No path found!")
            return -1, container



if __name__ == "__main__":
    xStart = 0
    yStart = 0
    zStart = 0
    xTarget = 49
    yTarget = 49
    zTarget = 4
    MAX_X = 50
    MAX_Y = 50
    MAX_Z = 5
    map = obstacle_map(xStart, yStart, zStart, xTarget, yTarget, zTarget, MAX_X, MAX_Y, MAX_Z)
    print(map)

    path, container = A_star_search(map, MAX_X, MAX_Y, MAX_Z)
    # A_star_search(map, MAX_X,MAX_Y)
    if path == -1:
        print("No path found!")
    else:
        path = np.array(path)  # 示例路径
    # visit_nodes = np.array([[1, 1, 1], [1, 2, 2], [0, 3, 3], [0, 4, 4]])  # 示例访问节点

    visualize_map_3d(map, path, container)

