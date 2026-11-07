class GlobalConfig:
    def __init__(self):
        """Configuration for SimLingo-LG Stage-2 Bench2Drive evaluation."""
        # -----------------------------------------------------------------------------
        # Lingo agent (unchanged controllers / dynamics)
        # -----------------------------------------------------------------------------
        self.eval_route_as = 'target_point' # "target_point" or "target_point_command" "command", -1 -> use from model config
        # if target_point_command (trained on both we eval with targetpoint)
        self.use_cot = True

        self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
        self.carla_fps = 20  # Simulator Frames per second
        self.stuck_threshold = 800
        self.creep_duration = 15  # Number of frames we will creep forward
        self.creep_throttle = 0.4
        self.inital_frames_delay = 2.0 / self.carla_frame_rate
        self.wp_dilation = 1  # Factor by which the wp are dilated compared to full CARLA 20 FPS
        self.data_save_freq = 5 # 5

        self.max_throttle = 1 # 0.75  # upper limit on throttle signal value in dataset
        self.brake_speed = 0.4  # desired speed below which brake is triggered
        # ratio of speed to desired speed at which brake is triggered
        self.brake_ratio = 1.1
        self.clip_delta = 1.0 # 0.25  # maximum change in speed input to longitudinal controller
        self.clip_throttle = 1.0 # 0.75  # Maximum throttle allowed by the controller

        # -----------------------------------------------------------------------------
        # PID controller
        # -----------------------------------------------------------------------------
        self.aim_distance_very_fast = 7.0  # 2.0
        self.aim_distance_fast = 3.0  # 2.0
        self.aim_distance_slow = 2.25
        self.aim_distance_threshold = 5.5
        self.aim_distance_threshold2 = 15
        self.turn_kp = 3.25 #1.25 -- these were the values for LB1
        self.turn_ki = 1.0 #0.75
        self.turn_kd = 1.0 #0.3
        self.turn_n = 20  # buffer size

        self.speed_kp = 1.75 #5.0
        self.speed_ki = 1.0 #0.5
        self.speed_kd = 2.0 #1.0
        self.speed_n = 20  # buffer size

        # -----------------------------------------------------------------------------
        # SIX RGB cameras: must match team_code/data_agent_surround.py
        # front, front_left, front_right, rear, rear_left, rear_right
        # -----------------------------------------------------------------------------
        self.num_cameras = [0, 1, 2, 3, 4, 5]
        for idx, yaw in enumerate((0.0, -60.0, 60.0, 180.0, -120.0, 120.0)):
            setattr(self, 'camera_pos_%d' % idx, [-1.5, 0.0, 2.0])
            setattr(self, 'camera_rot_%d' % idx, [0.0, 0.0, yaw])
            setattr(self, 'camera_width_%d' % idx, 1024)
            setattr(self, 'camera_height_%d' % idx, 512)
            setattr(self, 'camera_fov_%d' % idx, 110)
