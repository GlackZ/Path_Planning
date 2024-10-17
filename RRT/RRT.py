import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
np.random.seed(0)
random.seed(0)

class Tree:
    def __init__(self):
        self.list = []
    def add_node(self, x, y, xPre, yPre, dist, indPre):
        self.list.append([x, y, xPre, yPre, dist, indPre])
    def get_nearest_node(self, x, y):
        distance = np.inf
        nearest_node = None
        index = 0
        for i, node in enumerate(self.list):
            dist = np.sqrt((node[0] - x) ** 2 + (node[1] - y) ** 2)
            if dist < distance:
                distance = dist
                nearest_node = node
                index = i
        return index, nearest_node

class Env:
    def __init__(self):
        # 初始化整个空间，定义初始点、终点、采样点数、点与点之间的步长t等信息
        self.x_width = 25
        self.y_width = 12
        self.map = np.zeros((self.x_width, self.y_width))
        self.obstacle_list = self.obstacle_create()
        self.x0 = 6  # 定义初始点的x坐标
        self.y0 = 4  # 定义初始点的y坐标
        self.xn = 17  # 定义终点的x坐标
        self.yn = 5  # 定义终点的y坐标
        self.t = 0.1  # 点与点之间的步长
        self.Thr = 0.1  # 定义目标点与采样点之间的距离阈值
        self.map[self.x0][self.y0] = 4
        self.map[self.xn][self.yn] = 3
    def obstacle_create(self):
        obstacle_ratio = 0.2
        obstacle_list = []
        for i in range(self.x_width):
            for j in range(self.y_width):
                if np.random.rand() < obstacle_ratio:
                    self.map[i][j] = 1
                    obstacle_list.append([i, j])
        return np.array(obstacle_list)

    def show(self, rand_node_list = None, tree_list = None, path = None):
        plt.figure(figsize=(self.x_width/2, self.y_width/2))
        plt.xlim((-1, self.x_width))
        plt.ylim((-1, self.y_width))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(self.x_width))
        plt.yticks(np.arange(self.y_width))
        plt.grid()

        rand_node_list = np.array(rand_node_list)
        tree_list = np.array(tree_list)

        for i in range(self.x_width):
            for j in range(self.y_width):
                if self.map[i][j] == 1:
                    plt.scatter(i, j, color='black', s=300, marker='s')
                elif self.map[i][j] == 3:
                    plt.scatter(i, j, color='red', s=100)
                elif self.map[i][j] == 4:
                    plt.scatter(i, j, color='green', s=100)

        if rand_node_list is not None:
            plt.scatter(rand_node_list[:, 0], rand_node_list[:, 1], color='blue', s=50, marker='d')
        if tree_list is not None:
            plt.scatter(tree_list[:, 0], tree_list[:, 1], color='m', s=50, marker='x')
        if path is not None:
            plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2)

        plt.show()
        
# 在空间中随机产生一个点rand ->这个点不能是起点
def product_rand(tree, env):
    tree_set = set(map(tuple, np.array(tree.list)[:, 0:2]))
    while True:
        rand_node = [random.randint(0, env.x_width - 1) , random.randint(0, env.y_width - 1)]
        if tuple(rand_node) not in tree_set:
            return rand_node
        
def decide_direction(rand_node, nearest_node, env):
    z_value = np.sqrt((nearest_node[0] - rand_node[0]) ** 2 + (nearest_node[1] - rand_node[1]) ** 2)  # 斜边长度
    cos_value = (rand_node[0] - nearest_node[0]) / z_value
    sin_value = (rand_node[1] - nearest_node[1]) / z_value
    t = env.t
    x1 = nearest_node[0] + t * cos_value
    y1 = nearest_node[1] + t * sin_value
    if collision_check(nearest_node[0], nearest_node[1], x1, y1, env.obstacle_list):
        return x1, y1, True
    else:
        return nearest_node[0], nearest_node[1], False

def collision_check(x0, y0, x1, y1, obstacle_list):
    # 预计算线段的长度平方
    dx = x1 - x0
    dy = y1 - y0
    line_len_sq = dx ** 2 + dy ** 2
    
    # 如果线段非常短，直接判断起点和终点是否与障碍物重合
    if line_len_sq == 0:
        return all(np.hypot(x0 - x2, y0 - y2) >= 1 for x2, y2 in obstacle_list)
    
    for x2, y2 in obstacle_list:
        if not (x2 >= min(x0,x1)-1 and x2 <= max(x0,x1)+1 and y2 >= min(y0,y1)-1 and y2 <= max(y0,y1)+1):
            continue
        # 计算点到线的垂直距离
        d_line = abs(dy * x2 - dx * y2 + x1 * y0 - y1 * x0) / np.sqrt(line_len_sq)
        if d_line >= 0.7:
            continue
        # 计算垂足的参数 t
        t = ((x2 - x0) * dx + (y2 - y0) * dy) / line_len_sq
        
        if 0 <= t <= 1:
            # 如果垂足在线段上，使用垂直距离
            d = d_line
        elif t < 0:
            # 垂足在线段左侧，计算到起点的距离
            d = np.hypot(x0 - x2, y0 - y2)
        else:
            # 垂足在线段右侧，计算到终点的距离
            d = np.hypot(x1 - x2, y1 - y2)
        
        # 检查碰撞
        if d < 0.7:
            return False
    
    return True

def backtrace(tree:Tree, path:list):
    path.append(tree.list[-1][0:2])
    index_pre = tree.list[-1][5]
    while(True):
        path.append(tree.list[index_pre][0:2])
        print(tree.list[index_pre][0:2])
        if tree.list[index_pre][5] == -1:
            break
        else:
            index_pre = tree.list[index_pre][5]


if __name__ == '__main__':
    # print(error_list)
    env = Env()
    tree = Tree()
    tree.add_node(env.x0, env.y0, env.x0, env.y0, 0, -1)
    rand_node_list = []
    path = []

    while(True):
        rand_node = product_rand(tree, env)
        nearest_node_index, nearest_node = tree.get_nearest_node(rand_node[0], rand_node[1])
        x, y, find_new_node= decide_direction(rand_node, nearest_node, env)
        dist = sqrt((x - nearest_node[0]) ** 2 + (y - nearest_node[1]) ** 2)
        if find_new_node:
            tree.add_node(x, y, nearest_node[0], nearest_node[1], dist, nearest_node_index)
        dist_to_target = sqrt((x - env.xn) ** 2 + (y - env.yn) ** 2)
        if dist_to_target < env.Thr:
            backtrace(tree, path)
            path = np.array(path)
            print(path)
            break
        rand_node_list.append(rand_node)
    kwargs = {'rand_node_list': rand_node_list,
                  'tree_list': tree.list,
                  'path': path}
    env.show(**kwargs)
        # print(tree.list)
        # print(rand_node
    