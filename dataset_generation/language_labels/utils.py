import numpy as np
import cv2
import re
import xml.etree.ElementTree as ET
import json
import gzip

import simlingo_training.utils.transfuser_utils as t_u


def build_projection_matrix(w, h, fov):
    """

    函数作用:根据图像宽度、高度和相机视场角FOV,构建一个3x3的投影矩阵K(相机内参矩阵K),用于将3D点投影到2D图像平面上

    Build a projection matrix based on image dimensions and field of view.
    
    Args:
        w (int): Image width
        h (int): Image height
        fov (float): Field of view in degrees
    
    Returns:
        np.ndarray: 3x3 projection matrix
    """
    focal = w / (2.0 * np.tan(np.radians(fov / 2)))  # 焦距
    K = np.identity(3)  # 3✖3单位矩阵
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K
    # K为相机内参矩阵,它可以用于把相机坐标系中的3D点投影到图像坐标系中的2D点。K的形式如下:
    # [focal, 0, w/2]
    # [0, focal, h/2]
    # [0, 0, 1]
    # 其中focal是焦距,w/2和h/2是图像中心在图像坐标系中的位置。这个矩阵假设相机的主点在图像中心,并且没有径向畸变。 


def project_points(points2D_list, K):

    all_points_2d = []
    for point in  points2D_list:
        pos_3d = np.array([point[1], 0, point[0]])
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        points_2d, _ = cv2.projectPoints(pos_3d, 
                            rvec, tvec, 
                            K, 
                            dist_coeffs)
        all_points_2d.append(points_2d[0][0])
        
    return all_points_2d


def project_center_corners(obj, K):
    """
    Project the center corners of an object onto the image plane.
    
    Args:
        obj (dict): Object dictionary containing position, extent, and yaw
        K (np.ndarray): Projection matrix
    
    Returns:
        np.ndarray: 2D array of projected corner points
    """
    # check if obj is dict
    if isinstance(obj, dict):
        pos = obj['position']
        if 'extent' not in obj:
            extent = [0.15,0.15,0.15]
        else:
            extent = obj['extent']
        if 'yaw' not in obj:
            yaw = 0
        else:
            yaw = obj['yaw']
    else:
        # carla.BoundingBox
        pos = [obj.location.x, obj.location.y, obj.location.z]
        extent = [obj.extent.x, obj.extent.y, obj.extent.z]
        yaw = obj.rotation.yaw
        
    # get bbox corners coordinates
    corners = np.array([[-extent[0], 0, 0.75],
                        [extent[0], 0, 0.75]])

    # rotate bbox
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])
    corners = corners @ rotation_matrix.T

    # translate bbox
    corners = corners + np.array(pos)
    all_points_2d = []
    for corner in  corners:
        pos_3d = np.array([corner[1], -corner[2], corner[0]])
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        points_2d, _ = cv2.projectPoints(pos_3d, 
                            rvec, tvec, 
                            K, 
                            dist_coeffs)
        all_points_2d.append(points_2d[0][0])
        
    return np.array(all_points_2d)

def project_all_corners(obj, K):
    """
    Project all corners of a 3D bounding box onto the image plane.
    
    Args:
        obj (dict): Object dictionary containing position, extent, and yaw
        K (np.ndarray): Projection matrix
    
    Returns:
        tuple: (np.ndarray of 2D projected points, np.ndarray of 3D corner points)
    """
    """
    给定一个 3D 目标（中心位置 pos、半尺寸 extent、朝向 yaw），
    先在 3D 里构造 8 个角点 → 旋转 → ego坐标 → 再把每个角点投影到相机成像平面（像素坐标）。
    """
    pos = obj['position']      # 周围车在自车坐标系下的位置
    if 'extent' not in obj:    # 周围车辆的几何尺寸(长宽高,但是是一半)
        extent = [0.15,0.15,0.15]
    else:
        extent = obj['extent']
    if 'yaw' not in obj:
        yaw = 0
    else:
        yaw = obj['yaw']       # 周围车在自车坐标系下的航向角
            
    corners = np.array([
        [-extent[0], -extent[1], 0],  # 左后
        [extent[0], -extent[1], 0],   # 左前
        [extent[0], extent[1], 0],    # 右前
        [-extent[0], extent[1], 0],   # 右后
        [-extent[0], -extent[1], 2*extent[2]],   #
        [extent[0], -extent[1], 2*extent[2]],    #
        [extent[0], extent[1], 2*extent[2]],     #
        [-extent[0], extent[1], 2*extent[2]]     #
    ])   # 目标中心 pos 当作“底面中心”，而不是“几何中心”。

    # rotate bbox
    # 标准的绕 Z 轴旋转矩阵
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])
    corners = corners @ rotation_matrix.T

    # translate bbox
    corners = corners + np.array(pos)
    # 把 8 个角点从“局部坐标(这里指的是b_box的坐标系)”挪到自车所在坐标系中。
    # 到此为止，corners 是“自车三维坐标系(x向前y向右)”下的 8 个角点,从底部左后开始顺时针。
    all_points_2d = []
    for corner in  corners:  # 对每个 3D 角点做投影。
        pos_3d = np.array([corner[1], -corner[2], corner[0]]) # 这是相机坐标系x向右y向下z向前
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        points_2d, _ = cv2.projectPoints(pos_3d, 
                            rvec, tvec, 
                            K, 
                            dist_coeffs)
        all_points_2d.append(points_2d[0][0])
        
    return np.array(all_points_2d), np.array(corners)
    # all_points_2d 每个角点在图像坐标系中的像素坐标   8×2(底部:左后->左前->右前->右后 上部:左后->左前->右前->右后)
    # corners       每个角点在自车坐标系中的坐标      8×3


def is_vehicle_visible_in_image(vehicle_obj, MIN_X, MAX_X, MIN_Y, MAX_Y, CAMERA_MATRIX):
    """
    Check if a vehicle is visible in the image.
    """
    # Project the 3D points of the vehicle onto the 2D image plane
    projected_2d_points = project_center_corners(vehicle_obj, CAMERA_MATRIX)

    # Check if any projected point is visible
    vehicle_is_visible = False
    if projected_2d_points is None:
        return False

    for point_2d in projected_2d_points:
        if (point_2d[0] > MIN_X and point_2d[0] < MAX_X and
            point_2d[1] > MIN_Y and point_2d[1] < MAX_Y):
            vehicle_is_visible = True
            break

    return vehicle_is_visible


def should_consider_vehicle(vehicle, MIN_X, MAX_X, MIN_Y, MAX_Y, CAMERA_MATRIX):
    """
    True, if it's visible in the image and neither of the following applies
    False, if vehicle is not bicycle and the number of points on it are below a threshold
    False, if the vehicle is behind the ego vehicle
    False, if it's a parking vehicle, that does not cut in
    """
    """
    如果车辆在图像中可见且以下两种情况均不适用，则为真。
    如果车辆不是自行车且其上的点数低于阈值，则为假。
    如果车辆位于目标车辆后方，则为假。
    如果车辆是停放的车辆，则为假，这种情况不会发生。
    """
    # If the vehicle is parked and not cutting in, exclude it from consideration
    if vehicle['lane_type_str'] == "Parking" and not vehicle['vehicle_cuts_in']:
        return False
    # Max. distance is 25m and similar to the max. distance of junctions
    if  (
        vehicle['position'][0] < -1.5
        or (vehicle['base_type'] != 'bicycle' and vehicle['num_points'] < 15 and \
            'scenario' not in vehicle['role_name'])
        or ('scenario' in vehicle['role_name'] and vehicle['num_points'] < 10)
    ):
        return False

    # Check if the vehicle is visible in the image
    vehicle_is_visible = is_vehicle_visible_in_image(vehicle, MIN_X, MAX_X, MIN_Y, MAX_Y, CAMERA_MATRIX)

    return vehicle_is_visible

def light_state_to_word(light_state):
    '''
    0: NONE
    All lights off.
    1: Position
    2: LowBeam
    3: HighBeam
    4: Brake
    5: RightBlinker
    6: LeftBlinker
    7: Reverse
    8: Fog
    9: Interior
    10: Special1
    This is reserved for certain vehicles that can have special lights, like a siren.
    11: Special2
    This is reserved for certain vehicles that can have special lights, like a siren.
    12: All
    All lights on.

    '''
    light_state_dict = {0: 'None', 1: 'position light', 2: 'low beam', 3: 'high beam', 4: 'brake light', 5: 'right blinker',
                        6: 'left blinker', 7: 'reverse light', 8: 'fog light', 9: 'interior light', 10: 'emergency lights', 11: 'emergency lights',
                        12: 'All'}
    # add "the" in front of the light state
    light_state = light_state_dict[light_state]
    light_state = 'the ' + light_state
    return light_state

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)
    

def a_or_an(word):
    """
    Returns 'a' or 'an' depending on whether the word starts with a vowel or not.
    :param word: string
    :return: 'a' or 'an'
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    if word[0].lower() in vowels:
        return 'an'
    else:
        return 'a'

def number_to_word(number):
    """
    Returns the number as a word.
    :param number: int
    :return: string
    """
    number_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                   6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    return number_dict[number]

def get_scenario_name(measurement_file_current):
    # 示例(measurement_file_current): database/simlingo_v2_2026_02_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_532_route0_02_28_11_05_28/measurements/0032.json.gz
    """
    Extracts the scenario name from a given measurement file.
    Args:
        measurement_file_current (str): The path to the measurement file. (json or json.gz)
    Returns:
        str: The name of the scenario.
    The function performs the following steps:
    1. Opens and reads the measurement file.
    2. Extracts the route folder name from the file path using regex.
    3. Extracts the seed and route file number from the file path using regex.
    4. Determines whether the file is for training or validation.
    5. Constructs the path to the route file based on the extracted information.
    6. Extracts the route number from the file path using regex.
    7. Loads the route file and parses it to find the scenario name.
    8. If the route folder is 'custom_parkinglane', 'pedestrians', or 'OpensDoor', it directly assigns a scenario name.
    9. Otherwise, it parses the route file to find the closest scenario based on the current measurement's position.
    10. Returns the name of the closest scenario.
    """

    if measurement_file_current.endswith('.gz'):
        with gzip.open(measurement_file_current, 'r') as f:
            current_measurement = json.load(f)
    elif measurement_file_current.endswith('.json'):
        with open(measurement_file_current, 'r') as f:
            current_measurement = json.load(f)

    route_folder = measurement_file_current.split('simlingo/')[-1].split('/Town')[0]
    # 示例(route_folder): training_1_scenario/routes_training/random_weather_seed_1_balanced_150
    if route_folder is None:
        route_folder = re.search(r'custom_parkinglane', measurement_file_current)
        if route_folder is None:
            route_folder = re.search(r'pedestrians', measurement_file_current)
            if route_folder is None:
                route_folder = re.search(r'OpensDoor', measurement_file_current)

        try:
            route_folder = route_folder.group(0)
        except:
            print('not sure whihc scenario')
            exit()


    try:
        route_file = re.search(rf'Rep*(\d+)_*(\d+)_route*(\d+)', measurement_file_current).group(2)
        route_number = re.search(rf'Rep*(\d+)_*(\d+)_route*(\d+)', measurement_file_current).group(3)
        # 示例(route_file): 532  这是路线文件名,用于生成532.xml这种形式
        # 示例(route_number): 0  这是532.xml文件中的路线ID，也就是说此时的路线是532.xml文件中的route id="0"这条路线
    except:
        print(measurement_file_current)


    routefile_path = f'data/simlingo/{route_folder}/{route_file}.xml'  # 这是路线的xml文件的位置
    # 示例: /data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/532.xml

    # load route file
    if route_folder == 'custom_parkinglane':
        scenario_name = 'turn_in_parkinglane'
    elif route_folder == 'pedestrians':
        scenario_name = 'pedestrians'
    elif route_folder == 'OpensDoor':
        scenario_name = 'VehicleOpensDoor'
    else:
        tree = ET.parse(routefile_path)  # 使用 xml.etree.ElementTree 解析 XML 文件.
        # get route id=route_number
        root = tree.getroot()
        route_id = root.find(f'./route[@id="{route_number}"]')  # 这就是route id="0"这条路线的全部信息
        # get all information as dict
        route_info = {}
        locs = []
        scenarios = []
        for scenario in route_id.find('scenarios').iter('scenario'):
            p = scenario.find('trigger_point')  # 触发该场景的条件点, 当 ego 车靠近这个点时：这个 scenario 被激活,交通流开始生成
            loc = [float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])]  # 触发该场景的条件点的carla世界坐标
            loc_ego_coords = t_u.inverse_conversion_2d(np.array(loc[:2]), current_measurement['pos_global'], current_measurement['theta'])
            locs.append(loc_ego_coords)
            scenario_name = scenario.attrib['type']  # 交通流
            scenarios.append(scenario_name)

            route_info[scenario_name] = loc

        # current_loc = current_measurement['pos_global']
        # find the closest scenario
        # distances = [np.linalg.norm(np.array(loc[:2]) - np.array(current_loc)) for loc in locs]
        # behind_or_infront = [loc[0] > 0 for loc in locs]
        closest_behind = [np.linalg.norm(np.array(loc[:2])) if loc[0] < -10 else 999999 for loc in locs ]
        closest_scenario = scenarios[np.argmin(closest_behind)]
        scenario_name = closest_scenario
    
    return scenario_name


def is_pointing_towards_junction(ego, vehicle):
    orientation_relative_to_ego = vehicle['yaw']
    orientation_relative_to_ego = orientation_relative_to_ego * 180 / np.pi
    pos = vehicle['position']

    if ego['junction_id'] == -1 or vehicle['junction_id'] == -1:
        if pos[1] < -8 and orientation_relative_to_ego > 45 and orientation_relative_to_ego < 135:
            to_or_away_junction = "is pointing towards the junction"
            pointing_towards_junction = True
        elif pos[1] > 8 and orientation_relative_to_ego < -45 and orientation_relative_to_ego > -135:
            to_or_away_junction = "is pointing towards the junction"
            pointing_towards_junction = True
        elif pos[1] < -8 and orientation_relative_to_ego < -45 and orientation_relative_to_ego > -135:
            to_or_away_junction = "is pointing away from the junction"
            pointing_towards_junction = False
        elif pos[1] > 8 and orientation_relative_to_ego > 45 and orientation_relative_to_ego < 135:
            to_or_away_junction = "is pointing away from the junction"
            pointing_towards_junction = False
        elif pos[1] < 8 and pos[1] > -8 and orientation_relative_to_ego > 135 or orientation_relative_to_ego < -135:
            to_or_away_junction = "is pointing towards the junction"
            pointing_towards_junction = True
        elif pos[1] < 8 and pos[1] > -8 and orientation_relative_to_ego < 45 and orientation_relative_to_ego > -45:
            to_or_away_junction = "is pointing away from the junction"
            pointing_towards_junction = False
        else:
            to_or_away_junction = "is pointing in an unknown direction"
            pointing_towards_junction = False
        
    elif vehicle['next_junction_id'] == ego['next_junction_id'] or vehicle['next_junction_id'] == ego['junction_id']:
        pointing_towards_junction = True
        to_or_away_junction = "is pointing towards the junction"

    else:
        pointing_towards_junction = None
        to_or_away_junction = None

    return pointing_towards_junction, to_or_away_junction

def get_vehicle_appearance_string(object_box):

    if object_box['class'] == 'traffic_light':
        state_str = object_box['state']
        appearance_str = f'{state_str.lower()} traffic light'
    elif object_box['class'] == 'stop_sign':
        appearance_str = 'stop sign'
    elif object_box['class'] == 'static' and object_box['type_id']=='static.prop.trafficwarning':
        appearance_str = 'construction site'
    elif object_box['class'] == 'walker':
        if object_box["age"] == 'child':
            appearance_str = 'child'
        else:
            appearance_str = 'pedestrian'
    elif object_box['class'] == 'car':
        if object_box['position'][1] < 2 and object_box['position'][1] > -2:
            rough_pos_str = 'to the front'
        elif object_box['position'][1] > 2:
            rough_pos_str = 'to the front right'
        elif object_box['position'][1] < -2:
            rough_pos_str = 'to the front left'
        else:
            raise ValueError(f"Unknown position of vehicle {object_box['id']}.")
            
        if 'firetruck' in object_box['type_id']:
            vehicle_type = 'firetruck' # if random.random() < 0.5 else 'emergency vehicle'
        elif 'police' in object_box['type_id']:
            vehicle_type = 'police car' # if random.random() < 0.5 else 'emergency vehicle'
        elif 'ambulance' in object_box['type_id']:
            vehicle_type = 'ambulance' # if random.random() < 0.5 else 'emergency vehicle'
        elif 'jeep' in object_box['type_id']:
            vehicle_type = 'jeep'
        elif 'micro' in object_box['type_id']:
            vehicle_type = 'small car'
        elif 'nissan.patrol' in object_box['type_id']:
            vehicle_type = 'SUV'
        elif 'european_hgv' in object_box['type_id']:
            vehicle_type = 'HGV' # if random.random() < 0.5 else 'semi-truck'
        elif 'sprinter' in object_box['type_id']:
            vehicle_type = 'sprinter'
        else:
            vehicle_type = object_box['base_type']

        color_str = object_box["color_name"] + ' ' if object_box["color_name"] is not None and object_box["color_name"] != 'None' else ''
        if object_box['color_rgb'] == [0, 28, 0] or object_box['color_rgb'] == [12, 42, 12] or object_box['color_rgb'] == [0, 21, 0]:
            color_str = 'dark green '
        elif object_box['color_rgb'] == [0, 12, 58]:
            color_str = 'dark blue '
        elif object_box['color_rgb'] == [211, 142, 0]:
            color_str = 'yellow '
        elif object_box['color_rgb'] == [145, 255, 181]:
            color_str = 'blue '
        elif object_box['color_rgb'] == [215, 88, 0]:
            color_str = 'orange '

        appearance_str = f'{color_str}{vehicle_type} that is {rough_pos_str}'
    else:
        raise ValueError(f"Unknown object class {object_box['class']}.")

    return appearance_str