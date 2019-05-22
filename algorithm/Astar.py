env_data = [[3, 2, 2, 2, 2, 2, 2, 2, 1],
            [0, 0, 2, 2, 2, 2, 2, 0, 0],
            [2, 0, 0, 2, 2, 2, 0, 0, 2],
            [2, 2, 0, 0, 2, 0, 0, 2, 2],
            [2, 2, 2, 0, 0, 0, 2, 2, 2]]
			
"""
0: 普通格子（可通行）
1: 机器人的起点（可通行）
2: 障碍物（不可通行）
3: 宝藏箱（目标点）

"""		
			
##TODO 13 实现你的算法
#A-star寻路算法:
import heapq
class Node(object):
    def __init__(self, row_pos, column_pos, index, state, g = 0, h = 0):
        self.row_pos = row_pos
        self.column_pos = column_pos
        self.g = g
        self.h = h
        self.parent = None
        self.index = index #添加顺序，在f值相同的情况下，后添加的节点优先选取(f = g + h)
        self.state = state #节点的状态，0: 普通格子（可通行）1: 机器人的起点（可通行）2: 障碍物（不可通行）3: 宝藏箱（目标点）
    
    """
    估价公式：曼哈顿算法
    参数: cost横向和纵向的移动代价为 10 ，对角线的移动代价为 14 。
    之所以使用这些数据，是因为实际的对角移动距离是 2 的平方根，取整数好计算
    """
    def calcuateCostH(self, end_row_pos, end_column_pos):
        self.h = (abs(end_row_pos - self.row_pos) + abs(end_column_pos - self.column_pos)) * 10 
    
    def __lt__(self,other):#operator < 
        if (self.g + self.h) < (other.g + other.h):
            return True
        elif (self.g + self.h) > (other.g + other.h):
            return False
        else:
            return (self.index > other.index)
    
    def __eq__(self, other):
        return ((self.row_pos == other.row_pos) and (self.column_pos == other.column_pos))
   
    """
    优先级队列，始终弹出最小F值元素
    """
class PriorityQueue:
    def __init__(self):
        self.__queue = []
        self.__mapList = dict() #存储一个字典，以节点的行列(1,2)为key，value保存Node值.方便查询是否存在某一节点。和成员queue同步
   
    def push(self, node):
        heapq.heappush(self.__queue, node)
        self.__mapList[(node.row_pos, node.column_pos)] = node
    
    def pop(self):
        node = heapq.heappop(self.__queue)
        self.__mapList.pop((node.row_pos, node.column_pos))
        return node
    
    def isContainElement(self, row_pos, column_pos):
        if (row_pos,column_pos) in self.__mapList:
            return self.__mapList[(row_pos,column_pos)]
        else:
            return None
    
    def isEmpty(self):
        if len(self.__queue) == 0:
            return True
        else:
            return False
    
    def clear(self):
        self.__mapList.clear()
        self.__queue.clear()

def printPath(endNode):
    path_list = []
    while endNode:
        path_list.append(endNode)
        endNode = endNode.parent
    
    path_list = list(reversed(path_list))
    for item in path_list:
        print((item.row_pos,item.column_pos))
    

def searchAdjacentNode(next_Loc, currentNode, endNode, open_list, close_list, index, cost = 10):
    AdjNode_state = env_data[next_Loc[0]][next_Loc[1]]
    flag = True
    if AdjNode_state == 2:
        flag = False
    if close_list.isContainElement(next_Loc[0], next_Loc[1]):
        flag = False

    adjacent_node = open_list.isContainElement(next_Loc[0], next_Loc[1])
    if adjacent_node:
        if (currentNode.g + cost) < adjacent_node.g:
            adjacent_node.parent = currentNode
            adjacent_node.g = adjacent_node.parent.g + cost
        flag = False          
    else:
        adjacent_node = Node(next_Loc[0], next_Loc[1], index, AdjNode_state)
        adjacent_node.parent = currentNode
        adjacent_node.g = adjacent_node.parent.g + cost
        adjacent_node.calcuateCostH(endNode.row_pos, endNode.column_pos)
        open_list.push(adjacent_node)
    return flag
     
def findPathByAstar(start_node, end_node):
    if (start_node.row_pos < 0 or start_node.column_pos < 0 or start_node.row_pos >= rows 
        or start_node.column_pos >= columns or end_node.row_pos < 0 or end_node.column_pos < 0 or end_node.row_pos >= rows 
        or end_node.column_pos >= columns ):
        return
    
    if start_node.state == 2 or end_node.state == 2 :#障碍物无法通过
        return
    
    open_list = PriorityQueue()
    close_list = PriorityQueue()
    index = 0
    start_node.calcuateCostH(end_node.row_pos, end_node.column_pos)
    open_list.push(start_node)
    isFind = False
    while not open_list.isEmpty():
        minF_node = open_list.pop() #获取F值最小节点
        close_list.push(minF_node)
        if minF_node == end_node:
            isFind = True              
            printPath(minF_node)
            break

        #is_move_valid
        add_index_flag = False
        if minF_node.row_pos > 0:
            add_index_flag = searchAdjacentNode((minF_node.row_pos - 1, minF_node.column_pos), minF_node, end_node, open_list, 
                               close_list, index) #向上
            if add_index_flag:
                index += 1
        if minF_node.row_pos < rows - 1:
            add_index_flag = searchAdjacentNode((minF_node.row_pos + 1, minF_node.column_pos), minF_node, end_node, open_list, 
                               close_list, index) #向下
            if add_index_flag:
                index += 1       
        if minF_node.column_pos > 0:
            add_index_flag = searchAdjacentNode((minF_node.row_pos, minF_node.column_pos - 1), minF_node, end_node, open_list, 
                               close_list, index) #向左
            if add_index_flag:
                index += 1
        if minF_node.column_pos < columns - 1:
            add_index_flag = searchAdjacentNode((minF_node.row_pos, minF_node.column_pos + 1), minF_node, end_node, open_list, 
                               close_list, index) #向右
            if add_index_flag:
                index += 1
              
        if minF_node.row_pos > 0 and minF_node.column_pos > 0:
            add_index_flag = searchAdjacentNode((minF_node.row_pos - 1, minF_node.column_pos - 1), minF_node, end_node, open_list, 
                               close_list, index, 14) #左上
            if add_index_flag:
                index += 1
        if minF_node.row_pos < rows - 1 and minF_node.column_pos > 0:
            add_index_flag = searchAdjacentNode((minF_node.row_pos + 1, minF_node.column_pos - 1), minF_node, end_node, open_list, 
                               close_list, index, 14) #右下
            if add_index_flag:
                index += 1
        if minF_node.row_pos > 0 and minF_node.column_pos < columns - 1:
            add_index_flag = searchAdjacentNode((minF_node.row_pos - 1, minF_node.column_pos + 1), minF_node, end_node, open_list, 
                               close_list, index, 14) #右上
            if add_index_flag:
                index += 1
        if minF_node.row_pos < rows -1 and minF_node.column_pos < columns - 1:
            add_index_flag = searchAdjacentNode((minF_node.row_pos + 1, minF_node.column_pos + 1), minF_node, end_node, open_list, 
                               close_list, index, 14) #右下
            if add_index_flag:
                index += 1
                           
    if(isFind):
        print("找到宝藏！")
    else:
        print("没找到宝藏")
        
    open_list.clear()
    close_list.clear()
    

findPathByAstar(Node(0,8,0,1),Node(0,0,-1,3))