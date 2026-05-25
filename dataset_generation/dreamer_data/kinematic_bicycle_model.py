import numpy as np
# from config import GlobalConfig

class Config:
    def __init__(self, frame_rate=20):
        #  Time step for the model (20 frames per second).
        self.time_step = 1./frame_rate
        # Kinematic bicycle model parameters tuned from World on Rails.
        # Distance from the rear axle to the front axle of the vehicle.
        self.front_wheel_base = -0.090769015
        # Distance from the rear axle to the center of the rear wheels.
        self.rear_wheel_base = 1.4178275
        # Gain factor for steering angle to wheel angle conversion.
        self.steering_gain = 0.36848336
        # Deceleration rate when braking (m/s^2) of other vehicles.
        self.brake_acceleration = -4.952399
        # Acceleration rate when throttling (m/s^2) of other vehicles.
        self.throttle_acceleration = 0.5633837
        # Tuned parameters for the polynomial equations modeling speed changes
        # Numbers are tuned parameters for the polynomial equations below using
        # a dataset where the car drives on a straight highway, accelerates to
        # and brakes again
        # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
        self.throttle_values = np.array([9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02, -1.05996152e-03, 6.71079346e-04])
        # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
        self.brake_values = np.array([9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09, -4.91381935e-12])
        # Minimum throttle value that has an affect during forecasting the ego vehicle.
        self.throttle_threshold_during_forecasting = 0.3


# 运动学自从车模型
class KinematicBicycleModel():
    """
    Kinematic bicycle model describing the motion of a car given its state and action.
    Tuned parameters are taken from World on Rails.
    """

    def __init__(self, frame_rate=20):
        self.config = Config(frame_rate)

        self.time_step = self.config.time_step                  # 时间步长,模型每调用一次,就预测车辆在0.05s后的状态
        self.front_wheel_base = self.config.front_wheel_base    # 前轮基准距离参数
        self.rear_wheel_base = self.config.rear_wheel_base      # 后轮基准距离参数
        self.steering_gain = self.config.steering_gain          # 转向增益(模型输入的steer值(是[-1,1]之间的数值)乘以这个增益就是实际的转向角)
        self.brake_acceleration = self.config.brake_acceleration# 其他车辆刹车时使用的减速度，单位是 m/s²
        self.throttle_acceleration = self.config.throttle_acceleration  # 其他车辆踩油门时使用的加速度，单位是 m/s²
        self.throttle_values = self.config.throttle_values  # 自车油门速度预测的多项式系数
        self.brake_values = self.config.brake_values        # 自车刹车速度预测的多项式系数
        self.throttle_threshold_during_forecasting = self.config.throttle_threshold_during_forecasting  # 油门阈值,低于这个阈值时,预测自车速度时不再使用油门多项式模型

    
    
    # 一次预测N个其他车辆的未来状态,输入是其他车辆当前的状态和动作,输出是其他车辆未来的状态
    def forecast_other_vehicles(self, locations, headings, speeds, actions):
        """
        Forecast the future states of other vehicles based on their current states and actions.

        Args:
            locations (numpy.ndarray): Array of (x, y, z) coordinates representing the locations of other vehicles.
            headings (numpy.ndarray): Array of heading angles (in radians) for other vehicles.
            speeds (numpy.ndarray): Array of speeds (in m/s) for other vehicles.
            actions (numpy.ndarray): Array of actions (steer, throttle, brake) for other vehicles.

        Returns:
            tuple: A tuple containing the forecasted locations, headings, and speeds for other vehicles.
        """
        ####################  当前帧转向角、油门、刹车 ####################
        steers, throttles, brakes = actions[:, 0], actions[:, 1], actions[:, 2].astype(np.uint8)
        wheel_angles = self.steering_gain * steers  # 前轮转角
        slip_angles = np.arctan(self.rear_wheel_base / (self.front_wheel_base + self.rear_wheel_base) * np.tan(wheel_angles))  # 侧偏角
        
        ####################  下一帧 x y heading  speed ####################
        next_x = locations[:, 0] + speeds * np.cos(headings + slip_angles) * self.time_step
        next_y = locations[:, 1] + speeds * np.sin(headings + slip_angles) * self.time_step
        next_headings = headings + speeds / self.rear_wheel_base * np.sin(slip_angles) * self.time_step

        next_speeds = speeds + self.time_step * np.where(brakes, self.brake_acceleration, throttles * self.throttle_acceleration)
        next_speeds = np.maximum(0.0, next_speeds)

        next_locations = np.column_stack([next_x, next_y, locations[:, 2]])

        return next_locations, next_headings, next_speeds

    # 预测一个自车的未来状态,输入是自车当前的状态和动作,输出是自车未来的状态
    def forecast_ego_vehicle(self, location, heading, speed, action):
        """
        Forecast the future state of the ego vehicle based on its current state and action.

        Args:
            location (numpy.ndarray): Array of (x, y, z) coordinates representing the location of the ego vehicle.
            heading (float): Current heading angle (in radians) of the ego vehicle.
            speed (float): Current speed (in m/s) of the ego vehicle.
            action (numpy.ndarray): Action (steer, throttle, brake) for the ego vehicle.

        Returns:
            tuple: A tuple containing the forecasted location, heading, and speed for the ego vehicle.
        """
        steer, throttle, brake = action
        wheel_angle = self.steering_gain * steer  # 前轮转角
        slip_angle = np.arctan(self.rear_wheel_base / (self.front_wheel_base + self.rear_wheel_base) * np.tan(wheel_angle))  # 侧偏角
        
        ####################  下一帧 x y z heading  speed ####################
        next_x = (location[0] + speed * np.cos(heading + slip_angle) * self.time_step).item()
        next_y = (location[1] + speed * np.sin(heading + slip_angle) * self.time_step).item()
        next_z = location[2]
        next_heading = heading + speed / self.rear_wheel_base * np.sin(slip_angle) * self.time_step

        # We use different polynomial models for estimating the speed if whether the ego vehicle brakes or not.
        if brake:  # 刹车速度预测分支
            speed_kph = speed * 3.6  # 将速度从 m/s 转换为 km/h
            features = speed_kph ** np.arange(1, 8)  # 构造多项式特征 [speed_kph^1, speed_kph^2, ..., speed_kph^7], 这些特征用于估计刹车后的下一帧速度
            next_speed_kph = features @ self.brake_values
            next_speed = next_speed_kph / 3.6   # 下一帧速度(m/s)
        else:      # 油门速度预测分支
            throttle = np.clip(throttle, 0., 1.0)  # 把油门限制在 [0, 1] 范围内

            # For a throttle value < 0.3 the car doesn't really accelerate and the polynomial model below doesn't hold anymore.
            if throttle < self.throttle_threshold_during_forecasting:  # 当油门小于 0.3 时，车辆实际几乎不会加速，而且后面的多项式模型在这个区间不可靠,所以单独处理
                next_speed = speed.item()  # 下一帧速度(m/s)
            else:
                speed_kph = speed * 3.6
                features = np.array([speed_kph,
                                    speed_kph**2,
                                    throttle,
                                    throttle**2,
                                    speed_kph * throttle,
                                    speed_kph * throttle**2,
                                    speed_kph**2 * throttle,
                                    speed_kph**2 * throttle**2]).T

                next_speed_kph = features @ self.throttle_values
                next_speed = (next_speed_kph / 3.6).item()  # 下一帧速度(m/s)

        next_speed = np.array([np.maximum(0.0, next_speed)])
        next_location = np.array([next_x, next_y, next_z])

        return next_location, next_heading, next_speed
    

if __name__ == '__main__':
    from team_code.lateral_controller import LateralPIDController
    from team_code.longitudinal_controller import LongitudinalLinearRegressionController
    from team_code.config import GlobalConfig

    # test forecast_ego_vehicle
    ego_model = KinematicBicycleModel()
    config = GlobalConfig()
    _longitudinal_controller = LongitudinalLinearRegressionController(config)
    location = np.array([0., 0., 0.])
    heading = 0.
    speed = 20.

    target_speed = 10.
    action = np.array([0., 0., 0.0]) # steer, throttle, brake

    for _ in range(80):
        location, heading, speed = ego_model.forecast_ego_vehicle(location, heading, speed, action)
        print(location, heading, speed)

        throttle, control_brake = _longitudinal_controller.get_throttle_and_brake(False, target_speed, speed)
        action = np.array([0., throttle, control_brake])
        print(action)
