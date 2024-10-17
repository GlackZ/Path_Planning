import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
np.random.seed(0)
random.seed(0)
t = 0.1  # 点与点之间的步长
Thr = 0.1  # 定义目标点与采样点之间的距离阈值

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
    # 找到距离(x, y)小于3t的所有点
    def get_near_nodes(self, x, y, times=3):
        near_nodes = []
        indexs = []
        for i, node in enumerate(self.list):
            dist = np.sqrt((node[0] - x) ** 2 + (node[1] - y) ** 2)
            if dist < times*t:
                indexs.append(i)
                near_nodes.append(node)
        return indexs, near_nodes

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
        plt.figure(figsize=(self.x_width/1.5, self.y_width/1.5))
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
        plt.savefig('RRT_star_choose_min_parent_and_relink_100.png')
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
        # print(tree.list[index_pre][0:2])
        if tree.list[index_pre][5] == -1:
            break
        else:
            index_pre = tree.list[index_pre][5]

def relink(tree, new_node, env):
    near_indexs, near_nodes = tree.get_near_nodes(new_node[0], new_node[1], times = 100)
    for i in range(len(near_indexs)):
        dist = sqrt((new_node[0] - near_nodes[i][0]) ** 2 + (new_node[1] - near_nodes[i][1]) ** 2)
        if new_node[4] + dist < near_nodes[i][4] and collision_check(near_nodes[i][0], near_nodes[i][1], new_node[0], new_node[1], env.obstacle_list):
            near_nodes[i][4] = new_node[4] + dist
            near_nodes[i][5] = len(tree.list) - 1

if __name__ == '__main__':
    # print(error_list)
    env = Env()
    tree = Tree()
    tree.add_node(env.x0, env.y0, env.x0, env.y0, 0, -1)
    rand_node_list = []
    path = []

    while(True):
        # 产生一个随机点
        rand_node = product_rand(tree, env)
        # 找到距离rand_node最近的点
        nearest_node_index, nearest_node = tree.get_nearest_node(rand_node[0], rand_node[1])
        # 产生一个新的点，这个点是从nearest_node沿着rand_node方向走一步的点
        x, y, find_new_node= decide_direction(rand_node, nearest_node, env)

        ##### RRT* choose min parent #####
        if find_new_node:
            nearest_dist = 999999
            near_indexs, near_nodes = tree.get_near_nodes(x, y)
            for i in range(len(near_indexs)):
                dist = sqrt((x - near_nodes[i][0]) ** 2 + (y - near_nodes[i][1]) ** 2)
                if near_nodes[i][4] + dist < nearest_dist and collision_check(near_nodes[i][0], near_nodes[i][1], x, y, env.obstacle_list):
                    nearest_dist = near_nodes[i][4] + dist
                    nearest_node_index = near_indexs[i]
                    nearest_node = near_nodes[i]
            new_node = [x, y, nearest_node[0], nearest_node[1], nearest_dist, nearest_node_index]
            tree.add_node(new_node[0], new_node[1], new_node[2], new_node[3], new_node[4], new_node[5])
        ##### RRT* relink #####
            relink(tree, new_node, env)

        dist_to_target = sqrt((x - env.xn) ** 2 + (y - env.yn) ** 2)
        if dist_to_target < Thr:
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
    