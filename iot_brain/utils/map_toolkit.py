import re
import math 
import json
from collections import deque, defaultdict
from shapely.geometry import Polygon
import numpy as np
import heapq
from shapely.geometry import Polygon, Point
# --- 1. Core Data Structures (Unchanged) ---
class Node:
    """Represents a single point in the map with coordinates and tags."""
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.tags = {}

    def add_tag(self, k, v):
        self.tags[k] = v

    def get_id(self):
        """Generates a unique ID for the node for graph operations."""
        # A simple unique identifier based on coordinates.
        return f"node_{self.x}_{self.y}"

    def distance_to(self, other_node):
        """Calculates Euclidean distance to another Node."""
        if not isinstance(other_node, Node):
            raise ValueError("The other object must be an instance of Node.")
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)

    def __repr__(self):
        return f"Node({self.x}, {self.y}, Tags: {self.tags})"

class Location:
    """Represents a named area (like a room, hall, or building) containing nodes."""
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.topology = None # Can be Polygon for indoor or list of refs for outdoor
        self.cameras = []
        self.facilities = []
        self.elevators = []
        self.doors = []

    def add_node(self, node):
        self.nodes[node.get_id()] = node
        # Classify nodes upon adding
        tags_str = str(node.tags.values())
        if "surveillance" in tags_str: self.cameras.append(node)
        if "facility" in node.tags: self.facilities.append(node)
        if "elevator" in node.tags: self.elevators.append(node)
        if "door" in node.tags: self.doors.append(node)

    def add_topology(self, topology_data):
        self.topology = topology_data

    def __repr__(self):
        return (f"Location(Name: {self.name}, Nodes: {len(self.nodes)}, "
                f"Cameras: {len(self.cameras)}, Doors: {len(self.doors)})")

class NodeCollection:
    """A collection of all locations and nodes from a map file."""
    def __init__(self):
        self.locations = {}
        self.all_nodes_by_id = {}

    def add_location(self, location):
        self.locations[location.name] = location
        self.all_nodes_by_id.update(location.nodes)

    def get_location(self, location_name):
        return self.locations.get(location_name)
    
    def get_node_by_id(self, node_id):
        return self.all_nodes_by_id.get(node_id)
    def search_nodes(self, required_tags):
        """根据标签关键字查找符合条件的节点"""
        matched_nodes = []
        
        # 遍历所有位置及其节点
        for location in self.locations.values():
            for node in location.nodes.values():
                if node.matches_tags(required_tags):
                    matched_nodes.append(node)
        
        return matched_nodes

# --- 2. Map Parsers (Minor improvements for robustness) ---
def _parse_common(file_content: str) -> NodeCollection:
    node_collection = NodeCollection()
    # Use a more robust split pattern that handles various line endings
    location_blocks = re.split(r'\n\s*\n', file_content.strip())

    for block in location_blocks:
        if not block.strip():
            continue
        loc_name_match = re.search(r"<location_name:\s*(.*?)>", block)
        if not loc_name_match:
            continue
        
        location_name = loc_name_match.group(1).strip()
        location = Location(location_name)

        node_matches = re.findall(r"<node\s+x='(\d+)'\s+y='(\d+)'[^>]*>(.*?)</node>", block, re.DOTALL)
        for x, y, tag_block in node_matches:
            node = Node(x, y)
            tags = re.findall(r"<tag\s+k='(.*?)'\s+v='(.*?)'\s*/>", tag_block)
            for k, v in tags:
                node.add_tag(k, v)
            location.add_node(node)

        nd_refs = re.findall(r"<nd\s+ref='(.*?)'\s*/>", block)
        location.add_topology(nd_refs)
        node_collection.add_location(location)
    return node_collection

def get_indoor_nodes(osm_file: str):
    """
    解析室内场景的osm文件，并返回一个NodeCollection对象。
    此函数内部定义了其专属的数据结构，以避免与其他模块冲突。
    """

    # --- 专属于 get_indoor_nodes 的数据结构定义 ---
    class Node:
        def __init__(self, x, y):
            self.x = int(x)
            self.y = int(y)
            self.tags = {}

        def add_tag(self, k, v):
            self.tags[k] = v

        def matches_tags(self, required_tag):
            """检查节点的标签值中是否包含所有必需的标签值"""
            for i in self.tags.values():
                if required_tag in i:
                    return True
            return False

        def distance_to(self, other_node):
            if not isinstance(other_node, Node):
                raise ValueError("The other object must be an instance of Node.")
            return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)

        def __repr__(self):
            return f"Node({self.x}, {self.y}, Tags: {self.tags})"

    class Location:
        def __init__(self, name):
            self.name = name
            self.nodes = {}
            self.topology = None  # 用于存储单个Polygon对象
            self.cameras = []
            self.facilities = []
            self.elevators = []
            self.doors = []

        def add_node(self, node):
            self.nodes[(node.x, node.y)] = node

        def add_topology(self, polygon):
            self.topology = polygon  # 存储 Polygon 对象

        def add_camera(self, node):
            self.cameras.append(node)

        def add_facility(self, node):
            self.facilities.append(node)

        def add_elevator(self, node):
            self.elevators.append(node)

        def add_door(self, node):
            self.doors.append(node)

        def __repr__(self):
            return (f"Location(Name: {self.name}, Nodes: {len(self.nodes)}, Cameras: {len(self.cameras)}, "
                    f"Facilities: {len(self.facilities)}, Elevators: {len(self.elevators)}, Doors: {len(self.doors)})")

    class NodeCollection:
        def __init__(self):
            self.locations = {}

        def add_location(self, location):
            self.locations[location.name] = location

        def get_location(self, location_name):
            return self.locations.get(location_name)
        
        def search_nodes(self, required_tags):
            """根据标签关键字查找符合条件的节点"""
            matched_nodes = []
            for location in self.locations.values():
                for node in location.nodes.values():
                    if node.matches_tags(required_tags):
                        matched_nodes.append(node)
            return matched_nodes

    # --- 解析逻辑（您的代码，未做修改） ---
    node_collection = NodeCollection()

    try:
        with open(osm_file, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"错误：文件未找到，路径： {osm_file}")
        return node_collection # 返回一个空的集合对象

    # 按空行分割，处理每个location块
    location_blocks = re.split(r'\n\s*\n', file_content.strip())

    for location_block in location_blocks:
        if "<location_name:" in location_block:
            location_name_match = re.search(r"<location_name:\s*(.*?)>", location_block)
            if location_name_match:
                location_name = location_name_match.group(1).strip()
                location = Location(location_name)

                node_blocks = re.findall(r"<node\s+.*?>.*?</node>", location_block, re.DOTALL)
                nd_blocks = re.findall(r"<nd\s+ref='(.*?)'\s*/>", location_block)

                # 处理节点
                for block in node_blocks:
                    x_match = re.search(r"x='(\d+)'", block)
                    y_match = re.search(r"y='(\d+)'", block)
                    if x_match and y_match:
                        node_obj = Node(x_match.group(1), y_match.group(1))
                        tags = re.findall(r"<tag\s+k='(.*?)'\s+v='(.*?)'\s*/>", block)
                        for k, v in tags:
                            node_obj.add_tag(k, v)
                        
                        # 根据标签分类节点
                        if "surveillance" in node_obj.tags.values() and "elevator" not in node_obj.tags.keys():
                            location.add_camera(node_obj)
                        elif "facility" in node_obj.tags:
                            location.add_facility(node_obj)
                        elif "elevator" in node_obj.tags:
                            location.add_elevator(node_obj)
                        elif "door" in node_obj.tags:
                            location.add_door(node_obj)
                        location.add_node(node_obj)

                # 处理拓扑关系，构建Polygon
                # 为了从节点引用名找到节点坐标，先创建一个临时映射
                temp_node_map_by_tag = {val: node for node in location.nodes.values() for val in node.tags.values()}
                
                all_nd_coords = []
                for nd_str in nd_blocks:
                    node_refs = [ref.strip("' ") for ref in nd_str.split(',')]
                    for ref in node_refs:
                        if ref in temp_node_map_by_tag:
                            node = temp_node_map_by_tag[ref]
                            all_nd_coords.append((node.x, node.y))

                # 确保有足够的点来构建一个多边形
                if len(all_nd_coords) >= 3:
                    try:
                        polygon = Polygon(all_nd_coords)
                        location.add_topology(polygon)
                    except Exception as e:
                        # 如果坐标无法构成合法的Polygon（例如，自相交），则拓扑为空
                        print(f"警告：在 {location_name} 中创建Polygon失败: {e}")
                        location.add_topology(None)
                
                node_collection.add_location(location)

    return node_collection

def get_outdoor_nodes(osm_file: str):
    """
    解析室外场景的osm文件，并返回一个包含所有位置和节点的NodeCollection对象。
    该函数内部定义了其专属的数据结构，以避免与其它函数（如get_indoor_nodes）冲突。
    """

    # --- 数据结构定义 (在函数体内部) ---
    class Node:
        def __init__(self, x, y):
            self.x = int(x)
            self.y = int(y)
            self.tags = {}

        def add_tag(self, k, v):
            self.tags[k] = v

        def matches_tags(self, required_tag):
            """检查节点的标签值中是否包含所有必需的标签值"""
            for i in self.tags.values():
                if required_tag in i:
                    return True
            return False
            
        def distance_to(self, other_node):
            if not isinstance(other_node, Node):
                raise ValueError("The other object must be an instance of Node.")
            return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)

        def __repr__(self):
            return f"Node({self.x}, {self.y}, Tags: {self.tags})"

    class Location:
        def __init__(self, name):
            self.name = name
            self.nodes = {}
            self.topology = []
            self.cameras = []
            self.facilities = []
            self.elevators = []
            self.doors = []

        def add_node(self, node):
            self.nodes[(node.x, node.y)] = node

        def add_topology(self, node_refs):
            self.topology.append(node_refs)

        def add_camera(self, node):
            self.cameras.append(node)

        def add_facility(self, node):
            self.facilities.append(node)

        def add_elevator(self, node):
            self.elevators.append(node)

        def add_door(self, node):
            self.doors.append(node)

        def __repr__(self):
            return (f"Location(Name: {self.name}, Nodes: {len(self.nodes)}, "
                    f"Cameras: {len(self.cameras)}, Doors: {len(self.doors)})")

    class NodeCollection:
        def __init__(self):
            self.locations = {}

        def add_location(self, location):
            self.locations[location.name] = location

        def get_location(self, location_name):
            return self.locations.get(location_name)
        
        def search_nodes(self, required_tags):
            """根据标签关键字查找符合条件的节点"""
            matched_nodes = []
            for location in self.locations.values():
                for node in location.nodes.values():
                    if node.matches_tags(required_tags):
                        matched_nodes.append(node)
            return matched_nodes

    # --- 解析逻辑 (您的代码，未作修改) ---
    node_collection = NodeCollection()

    with open(osm_file, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # 将文本按换行符分割为块，处理每个块
    location_blocks = file_content.split('\n\n')

    for location_block in location_blocks:
        if "<location_name:" in location_block:
            # 提取位置名
            location_name_match = re.search(r"<location_name:\s*(.*?)>", location_block)
            if location_name_match:
                location_name = location_name_match.group(1).strip()
                location = Location(location_name)

                # 匹配每个<node>块
                node_blocks = re.findall(r"<node\s+.*?>.*?</node>", location_block, re.DOTALL)

                # 匹配每个nd块
                nd_blocks = re.findall(r"<nd\s+ref='(.*?)'\s*/>", location_block)

                # 处理节点
                for block in node_blocks:
                    x_match = re.search(r"x='(\d+)'", block)
                    y_match = re.search(r"y='(\d+)'", block)

                    if x_match and y_match:
                        x = int(x_match.group(1))
                        y = int(y_match.group(1))
                        node_obj = Node(x, y)

                        # 匹配标签
                        tags = re.findall(r"<tag\s+k='(.*?)'\s+v='(.*?)'\s*/>", block)
                        for k, v in tags:
                            node_obj.add_tag(k, v)

                        # 根据标签分类
                        if "surveillance" in node_obj.tags.values():
                            location.add_camera(node_obj)
                        elif "facility" in node_obj.tags.keys():
                            location.add_facility(node_obj)
                        elif "elevator" in node_obj.tags.keys():
                            location.add_elevator(node_obj)
                        elif "door" in node_obj.tags.keys():
                            location.add_door(node_obj)

                        location.add_node(node_obj)

                # 处理拓扑关系
                location_topology = []
                for nd in nd_blocks:
                    location_topology.append(nd.split(', '))

                location.add_topology(location_topology)

                # 将位置添加到NodeCollection
                node_collection.add_location(location)

    return node_collection

def bfs_all_paths(graph, start_list, goal_list):
    # 用于存储所有找到的路径
    all_paths = []

    # 遍历 start 和 goal 列表中的每一对
    for start in start_list:
        for goal in goal_list:
            # 用于存储当前 start 到 goal 的路径
            paths = []
            # 队列存储路径，每个路径是一个节点的列表
            queue = deque([[start]])
            # 用于记录已经访问过的节点，避免死循环
            visited = set([start])

            while queue:
                # 取出队列中的第一个路径
                path = queue.popleft()
                node = path[-1]

                # 如果到达目标节点，记录路径
                if node == goal:
                    paths.append(path)
                    continue  # 继续搜索其他路径

                # 遍历当前节点的邻居
                for neighbor in graph.get(node, []):
                    # 如果邻居没有访问过，且不在当前路径中（避免环路）
                    if neighbor not in visited:
                        # 标记为已访问
                        visited.add(neighbor)
                        # 将新的路径加入队列
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)

            # 将当前找到的路径添加到总路径集合
            all_paths.extend(paths)

    return all_paths


def optional_road_path_search(nodecollection, origin_location, destination_location,osm_file='./connectivity.txt'):
    # 解析连接关系并构建图
    road_connections = defaultdict(list)
    
    with open(osm_file, 'r') as file:
        road_data = file.read().strip().split('\n')
        
    # 遍历每行，解析道路与其连接关系
    road = None
    for line in road_data:
        if line.startswith("Road:"):
            # 提取道路名称
            road = line.split(':')[1].strip()
        elif line.startswith("Connectivity:"):
            # 提取连接的道路
            connections = line.split(':')[1].strip().split('&')
            for conn in connections:
                road_connections[road].append(conn.strip())
                # 确保双向连接
                road_connections[conn.strip()].append(road)  # 反向连接
    origin_doors = []
    dest_doors = []
    for door in nodecollection.get_location(origin_location).doors:
        origin_doors.append(door.tags['door'])
    for door in nodecollection.get_location(destination_location).doors:
        dest_doors.append(door.tags['door'])
    
    # 查找从origin_location到destination_location的所有路径
    all_paths = bfs_all_paths(road_connections, origin_doors, dest_doors)


    return all_paths  # 返回路径列表
def complete_path(nodes,path):
        """
        补全路径中的端点，根据相邻节点的公共标签，推断出缺失的端点。
        
        :param path: 路径端点列表，包含Node对象
        :return: 补全后的路径端点列表
        """
        complete_path = [path[0]]  # 初始化路径，包含第一个节点
    
        # 遍历相邻节点对，补全路径
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
    
            # 获取公共标签
            common_tags = set(current_node.tags.keys()) & set(next_node.tags.keys())
    
            for common_tag in common_tags:
                # 获取公共标签对应的端点值
                current_value = current_node.tags[common_tag]
                next_value = next_node.tags[common_tag]
    
                # 提取端点编号（假设格式为 tag_value_编号）
                current_index = int(re.search(r'(\d+)', current_value).group(1))
                next_index = int(re.search(r'(\d+)', next_value).group(1))
    
                # 判断端点的升序还是降序
                if current_index < next_index:
                    # 按升序补充中间端点
                    for index in range(current_index + 1, next_index):
                        intermediate_tag = f"{common_tag}_{index}"
                        intermediate_node = nodes.search_nodes(intermediate_tag)[0]
                        complete_path.append(intermediate_node)
                elif current_index > next_index:
                    # 按降序补充中间端点
                    for index in range(current_index - 1, next_index, -1):
                        intermediate_tag = f"{common_tag}_{index}"
                        intermediate_node = nodes.search_nodes(intermediate_tag)[0]
                        complete_path.append(intermediate_node)
    
            # 将下一个节点加入到路径中
            complete_path.append(next_node)
    
        return complete_path
def remove_duplicate_nodes(node_list):
    """
    去除节点列表中的重复节点。认为两个节点重复的条件是：
    它们的 x、y 坐标相同，并且 tags 中的键值对完全相同（不考虑字典的顺序）。
    :param node_list: 节点对象列表
    :return: 去重后的节点列表
    """
    seen = set()
    unique_nodes = []
    for node in node_list:
        # 使用节点的坐标和排序后的 tags 项生成一个唯一键
        key = (node.x, node.y, tuple(sorted(node.tags.items())))
        if key not in seen:
            seen.add(key)
            unique_nodes.append(node)
    return unique_nodes

def road_path_nodes_search(nodes, road_path):
    """
    获取路径中的所有端点集合
    :param nodes: NodeCollection 对象
    :param road_path: 路段列表，例如 [['playground_back_door', 'Ankang_road', 'Yongqing_side_road_2', 'Yongqing_road']]
    :return: 端点集合的节点列表
    """
    
    all_endpoints = []

    # 遍历路径中的每个路段
    # for path in road_path:
        # 初始化路径的端点
    path_endpoints = []

        # 遍历路径中的每一段路
    for i in range(len(road_path) - 1):
        start_segment = road_path[i]
        end_segment = road_path[i + 1]
            # 搜索同时包含 start_segment 和 end_segment 的节点
        ver1_lst = remove_duplicate_nodes(nodes.search_nodes(start_segment))

        ver2_lst = remove_duplicate_nodes(nodes.search_nodes(end_segment))

        lst = []
        for node1 in ver1_lst:
            for node2 in ver2_lst:
                if node1.x == node2.x and node1.y == node2.y and 'surveillance' not in node1.tags.values():
                    lst.append(node1)

        path_endpoints.append(lst[0])
        # 将这条路径的端点添加到总端点列表
        path_endpoints = complete_path(nodes,path_endpoints)
        all_endpoints.extend(path_endpoints)
    
    return remove_duplicate_nodes(all_endpoints)
def road_path_length_calculation(path_lst):
    total_dist = 0
    for i in range(len(path_lst)-1):
        total_dist += path_lst[i].distance_to(path_lst[i+1])
    return total_dist

def best_path(node,path_lst):
    optional_path = path_lst
    best_len = 1000000
    best_index = -1
    for i in range(len(optional_path)):
        path_node_lst = road_path_nodes_search(node,optional_path[i])
        if road_path_length_calculation(path_node_lst) < best_len:
            best_len = road_path_length_calculation(path_node_lst)
            best_index = i
    return road_path_nodes_search(node,optional_path[best_index])

# 计算欧几里得距离
def euclidean_distance(node1, node2):
    return np.linalg.norm(np.array(node1) - np.array(node2))

# 判断节点是否在建筑物内
def is_in_building(point, buildings):
    return any(building.contains(Point(point)) for building in buildings)

# 获取位置的门节点
def get_doors_for_location(location):
    return [node for node in location.nodes.values() if 'door' in node.tags][0]

# A*算法：计算从start到end的最短路径，优化搜索策略
def a_star(start, end, all_nodes, buildings, grid_size=500, tolerance=20, dynamic_step=False):
    open_list = []
    closed_list = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, end)}

    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        current = heapq.heappop(open_list)[1]

        # 终止条件：如果当前节点距离目标节点小于等于容忍误差（例如20米）
        if euclidean_distance(current, end) <= tolerance:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_list.add(current)

        # 获取邻居节点，考虑上下左右和四个斜向邻居，扩大步长
        neighbors = [
            (current[0] - 3, current[1]),  # 上
            (current[0] + 3, current[1]),  # 下
            (current[0], current[1] - 3),  # 左
            (current[0], current[1] + 3),  # 右
            (current[0] - 3, current[1] - 3),  # 左上
            (current[0] - 3, current[1] + 3),  # 右上
            (current[0] + 3, current[1] - 3),  # 左下
            (current[0] + 3, current[1] + 3)   # 右下
        ]

        for neighbor in neighbors:
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in closed_list:
                if is_in_building(neighbor, buildings):
                    continue

                # 增加步长，减少计算量
                step_cost = 3  # 增加步长为3

                tentative_g_score = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + euclidean_distance(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # 没有路径
def calculate_distance(camera, path_point):
    camera_point = Point(camera.x, camera.y)
    path_point = Point(path_point[0], path_point[1])
    return camera_point.distance(path_point)

# 获取符合条件的location
def get_hall_locations(nodecollection):
    hall_locations = []
    for location_name, location in nodecollection.locations.items():
        # 过滤包含hall但不包含conference_hall, lecture_hall, assembly_hall的location
        if 'hall' in location_name.lower() and not any(exclude in location_name.lower() for exclude in ['conference-hall', 'lecture-hall', 'assembly-hall','presentation-hall','rehearsal-hall']):
            hall_locations.append(location)
    return hall_locations