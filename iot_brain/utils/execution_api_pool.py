from . import map_toolkit # Ensure this import is correct based on your project structure
# This file contains the API toolkit for verifying various aspects of campus maps.
import json
from collections import defaultdict
import re
import math
from shapely.geometry import Polygon
from pathlib import Path
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD
class ExercutionApiPool:

    # 1. **【关键修改】** 构造函数现在接收一个 `topology_data_path`
    def __init__(self, topology_data_path: str | Path):
        """
        初始化API池。
        
        Args:
            topology_data_path (str | Path): 指向 `topology_sample` 文件夹的路径。
        """
        self.topology_path = Path(topology_data_path)
        self._cache = {}  # 用于缓存已加载的NodeCollection对象: {'map_name': NodeCollection}
    def _get_node_collection(self, map_name: str):
        """
        【内部方法】加载或从缓存中获取地图数据。
        
        Args:
            map_name (str): 地图的名称，如 'campus' 或 'library_1F'。
        
        Returns:
            map_toolkit.NodeCollection: 加载好的数据对象。
        """
        if map_name in self._cache:
            return self._cache[map_name]

        # 根据地图名称判断是调用室内还是室外解析器
        if map_name == 'campus':
            file_path = self.topology_path / 'campus.txt'
            collection = map_toolkit.get_outdoor_nodes(str(file_path))
        else:
            # 室内地图文件命名约定：teaching_building_1_1F.txt
            file_name = f"{map_name.replace(' ', '_').lower()}.txt"
            file_path = self.topology_path / file_name
            collection = map_toolkit.get_indoor_nodes(str(file_path))
        
        if not collection.locations:
             raise FileNotFoundError(f"地图文件加载失败或为空: {file_path}")

        self._cache[map_name] = collection
        return collection
    
    def road_path_trajectory_fitting(self, start_location, dest_location, start_point=None, dest_point=None):
        """
        【API 1】室外路径拟合。
        在室外场景中，根据起点、终点等信息拟合最佳路径。
        """
        # 这个API只适用于室外'campus'地图
        start_location = start_location.replace('_', '-').lower()
        dest_location = dest_location.replace('_', '-').lower()
        campus_collection = self._get_node_collection('campus')
        connectivity_file = str(self.topology_path / 'connectivity.txt')
        # 获取所有可选的路径列表
        origin_path_lst = map_toolkit.optional_road_path_search(campus_collection, start_location, dest_location, osm_file=connectivity_file)

        # 定义辅助函数，通过遍历节点的 tags 值，判断是否含有目标名称
        def node_has_name(node, target_name):
            return any(target_name == val for val in node.tags.values())
        
        # 如果指定了起始点名称，过滤所有起点与 start_point 不匹配的路径
        if start_point is not None:
            origin_path_lst = [path for path in origin_path_lst if node_has_name(path[0], start_point)]
        
        # 如果指定了终点名称，过滤所有终点与 dest_point 不匹配的路径
        if dest_point is not None:
            origin_path_lst = [path for path in origin_path_lst if node_has_name(path[-1], dest_point)]
        
        # 如果过滤后没有候选路径，则抛出异常或根据需求返回空列表
        if not origin_path_lst:
            raise ValueError("没有找到符合条件的路径，请检查起始点和终点名称是否正确")
        
        # 调用 best_path 方法，选择过滤后的最优路径
        best_trajectory = map_toolkit.best_path(campus_collection, origin_path_lst)
        
        return best_trajectory
    def road_path_camera_search(self, endpoint_lst: list):
        campus_collection = self._get_node_collection('campus')
        camera_nodes = []  # 用于存储最终匹配到的摄像头节点

        # 按照每个路径段进行分组处理
        path_segments = {}  # 存储每个路径段的节点（根据标签名分组）
        for node in endpoint_lst:
            for tag_key in node.tags.keys():
                if '_vertex' in tag_key:
                    path_segments.setdefault(tag_key, []).append(node)

        # 遍历每个路径段
        for key, nodes in path_segments.items():
            # 提取路径的x、y范围
            min_x = min(node.x for node in nodes)
            max_x = max(node.x for node in nodes)
            min_y = min(node.y for node in nodes)
            max_y = max(node.y for node in nodes)

            # 去掉 '_vertex' 获取路径名称
            location_name = key.replace('_vertex', '').replace('_', '-')

            # 获取与该路径名称相关的location对象
            location = campus_collection.get_location(location_name)

            if location:
                # 获取该路段上的所有摄像头节点
                for camera in location.cameras:
                    # 检查摄像头是否在路径的坐标范围内
                    if min_x <= camera.x <= max_x and min_y <= camera.y <= max_y:
                        camera_nodes.append(camera)

        return camera_nodes
    def scenario_object_location(self, location_name: str, scenario_name: str, object_name: str):
        scenario_name = scenario_name.replace('_', '-').lower()
        # This function is correct and remains unchanged.
        file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
        collection = map_toolkit.get_indoor_nodes(file_path)
        location_obj = collection.get_location(scenario_name)
        target_node = None
        for node in location_obj.nodes.values():
            # 假设目标参照物是节点的标签之一
            if object_name in node.tags.values():
                target_node = node
                break
        
        if not target_node:
            return f"Object {object_name} not found in location {scenario_name}."

        # 计算目标参照物和所有摄像头的欧式距离，找到距离最近的摄像头
        min_distance = float('inf')
        closest_camera = None

        for camera in location_obj.cameras:
            # 计算欧式距离
            distance = math.sqrt((target_node.x - camera.x) ** 2 + (target_node.y - camera.y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_camera = camera

        if closest_camera:
            print(f"Closest camera to {object_name} at {location_name}: Camera {closest_camera} with distance {min_distance:.2f}")
            return closest_camera
        else:
            return f"No cameras found in location {location_name}."
    
    def camera_coverage_search(self, location_name: str, scenario_name: str):
        scenario_name = scenario_name.replace('_', '-').lower()
        # 获取指定位置
        file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
        collection = map_toolkit.get_indoor_nodes(file_path)
        location_obj = collection.get_location(scenario_name)
        
        if not location_obj:
            return f"Location {scenario_name} not found."
        
        # 获取所有节点坐标
        nodes = list(location_obj.nodes.values())

        # 计算最大横向和纵向距离，用于确定摄像头的感知半径
        max_x_distance = max(node.x for node in nodes) - min(node.x for node in nodes)
        max_y_distance = max(node.y for node in nodes) - min(node.y for node in nodes)
        coverage_radius = 0.3 * max(max_x_distance, max_y_distance)
        
        # 获取位置内的所有摄像头
        cameras = location_obj.cameras

        # 每个摄像头可以覆盖的节点集合
        camera_coverage = {}
        for camera in cameras:
            camera_coverage[camera] = []
            for node in nodes:
                # 计算欧氏距离
                distance = math.sqrt((camera.x - node.x) ** 2 + (camera.y - node.y) ** 2)
                if distance <= coverage_radius:
                    camera_coverage[camera].append(node)
        
        # 建立集合覆盖ILP问题：目标为最小化所选摄像头的数量
        prob = LpProblem("Minimum_Cameras_for_Cover", LpMinimize)
        
        # 为每个摄像头定义二元选择变量（0表示不选，1表示选择）
        camera_vars = LpVariable.dicts("Camera", cameras, 0, 1, cat="Binary")
        
        # 目标函数：最小化摄像头的选取数量
        prob += lpSum([camera_vars[camera] for camera in cameras]), "Minimize_Number_of_Cameras"
        
        # 对每个节点添加约束：如果节点可以被至少一个摄像头覆盖，则必须至少有一个摄像头被选择
        for node in nodes:
            # 找出所有可以覆盖该节点的摄像头
            covering_cams = [camera for camera in cameras if node in camera_coverage[camera]]
            if covering_cams:
                prob += lpSum([camera_vars[c] for c in covering_cams]) >= 1, f"Cover_Node_{node.x}_{node.y}"
            else:
                # 若某个节点无法被任何摄像头覆盖，这里可根据需求采取措施（例如记录该节点）
                pass
        
        # 求解ILP问题
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # 获取被选中的摄像头
        selected_cameras = [camera for camera in cameras if camera_vars[camera].varValue == 1]
        
        # 如果没有选中任何摄像头（理论上不应该发生，因为约束要求至少有一个摄像头覆盖每个节点）
        if not selected_cameras:
            if cameras:
                selected_cameras = [cameras[0]]
            else:
                return []
        
        return selected_cameras

    def indoor_path_search(self, location_name: str, location_with_door_dict: dict, entrance_door: str = None, dynamic_step=True):
        file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
        collection = map_toolkit.get_indoor_nodes(file_path)

        
        buildings = []
        doors = {}
        all_nodes = {}
        location_list = list(location_with_door_dict.keys())


        # 获取所有位置的节点、门节点和拓扑结构
        for location_name in location_list:
            location = collection.get_location(location_name)
            if location_with_door_dict[location_name] != None:
                doors[location_name] = collection.search_nodes(location_with_door_dict[location_name])[0]
            else:
                doors[location_name] = map_toolkit.get_doors_for_location(location)
        if entrance_door != None:
            location_list.insert(0,'start')
            doors['start'] = collection.search_nodes(entrance_door)[0]
        # print(location_list)
        all_location = list(collection.locations.keys())
        index = 1000
        for i in range(len(all_location)):
            if 'hall' in all_location[i].lower() and not any(exclude in all_location[i].lower() for exclude in ['conference-hall', 'lecture-hall', 'assembly-hall','laboratory-area-hall','office-area-hall','conference-hall', 'lecture-hall', 'assembly-hall','presentation-hall','rehearsal-hall']):
                index = i
        if index != 1000:
            del all_location[index]
        for location_name in all_location:
            location = collection.get_location(location_name)
            buildings.append(Polygon(location.topology))  # 存储位置的拓扑边界
            all_nodes[location_name] = [(int(x), int(y)) for x, y in collection.get_location(location_name).topology.exterior.coords]

        # 存储路径
        path = []

        # 计算每两两location之间的路径
        for i in range(len(location_list) - 1):
            start_location = location_list[i]
            end_location = location_list[i + 1]

            # 获取最近的door节点
            start_door = doors[start_location]
            end_door = doors[end_location]


            # 传递坐标给a_star（确保是(x, y)坐标而不是Node对象）
            start_coords = (start_door.x, start_door.y)
            end_coords = (end_door.x, end_door.y)

            # 获取路径
            path_segment = map_toolkit.a_star(start_coords, end_coords, all_nodes, buildings, dynamic_step=dynamic_step)
            if path_segment:
                path.extend(path_segment)

        return path

    def indoor_path_camera_search(self, location_name: str, path_list: list):
        file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
        collection = map_toolkit.get_indoor_nodes(file_path)
        # 获取符合条件的hall位置
        hall_locations = map_toolkit.get_hall_locations(collection)
        location = hall_locations[0]
        monitored_cameras = []
        # 获取最大横向和纵向的距离
        nodes = list(location.nodes.values())
        max_x_distance = max(node.x for node in nodes) - min(node.x for node in nodes)
        max_y_distance = max(node.y for node in nodes) - min(node.y for node in nodes)
        
        # 计算摄像头的感知半径，设定为最大距离的30%
        coverage_radius = 0.3 * max(max_x_distance, max_y_distance)
        for camera in hall_locations[0].cameras:
            for path_point in path_list:
                # 计算摄像头与路径点的距离
                distance = map_toolkit.calculate_distance(camera, path_point)
                if distance <= coverage_radius:
                    # 如果摄像头可以监控到路径点，添加该摄像头到结果列表
                    monitored_cameras.append(camera)
                    break  # 不需要继续检查此摄像头，跳出循环

        
        return monitored_cameras
