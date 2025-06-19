import rclpy, math, time
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformException, TransformListener
import numpy as np


class DWAPathPlanner(Node):
    def __init__(self):
        super().__init__('dwa_path_planner')

        # 初始化參數
        self.obstacle_weight = 0.28  # 障礙物權重
        self.goal_weight = 0.7     # 目標點權重
        self.velocity_weight = 0.02 # 速度權重
        
        self.global_frame = 'odom'
        self.robot_frame = 'rplidar_link'
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        tf_ready = self.wait_for_tf('odom', 'rplidar_link', timeout_sec=3.0)
        # 發佈路徑
        self.DWA_path_publisher = self.create_publisher(MarkerArray, '/dwa_paths', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        if tf_ready:
            self.cmd_vel_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.cmd_vel_callback,
                QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
            )
        else:
            self.get_logger().error("TF not available. Node will not publish cmd_vel.")

        self.angle_subscriber = self.create_subscription(
            Marker,
            # '/robot_to_leg_vector',
            '/robot_to_following_position_vector',
            self.vector_arrows_callback,
            10)
        # 訂閱雷射資料
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.leg_center_subscriber = self.create_subscription(
            Marker,
            '/leg_center',
            self.leg_center_callback,
            10)
        self.laser_data = None
        self.current_velocity = Twist()
        self.timer = self.create_timer(0.1, self.generate_paths)

        self.target_point = Point(x=0.0, y=0.0, z=0.0) 

        self.best_path_points = None 
        self.best_path_index = -1
        self.best_path_velocity = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    def wait_for_tf(self, target_frame, source_frame, timeout_sec=3.0):
        start_time = self.get_clock().now()
        self.get_logger().info(f"Waiting for TF transform from [{source_frame}] to [{target_frame}]...")

        while rclpy.ok():
            try:
                now = rclpy.time.Time(seconds=0)
                self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    now,
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
                self.get_logger().info("TF transform is available. Proceeding...")
                return True
            except Exception as e:
                current_time = self.get_clock().now()
                elapsed = (current_time.nanoseconds - start_time.nanoseconds) / 1e9
                if elapsed > timeout_sec:
                    self.get_logger().warn(f"Timeout waiting for TF from [{source_frame}] to [{target_frame}]: {e}")
                    return False
                rclpy.spin_once(self, timeout_sec=0.1)
    def leg_center_callback(self, msg):
        if len(msg.points) == 0:
            return

        current_time = self.get_clock().now()
        lidar_frame = 'rplidar_link'
        target_frame = 'odom'  

        # 轉換成 PointStamped 才能用 tf2 做座標轉換
        raw_point = PointStamped()
        raw_point.header.frame_id = lidar_frame
        rclpy.time.Time(seconds=0).to_msg()
        raw_point.point = msg.points[0]

        try:
            transformed_point = self.tf_buffer.transform(raw_point, target_frame, timeout=rclpy.duration.Duration(seconds=0.5))
            current_pos = transformed_point.point
            current_time_sec = current_time.nanoseconds / 1e9

            if hasattr(self, 'last_leg_center') and hasattr(self, 'last_leg_time'):
                dt = current_time_sec - self.last_leg_time
                if dt > 0:
                    dx = current_pos.x - self.last_leg_center.x
                    dy = current_pos.y - self.last_leg_center.y
                    distance = math.hypot(dx, dy)
                    self.target_speed = distance / dt
                    # self.get_logger().info(f"[leg_center] Estimated target speed (world frame): {self.target_speed:.2f} m/s")

            self.last_leg_center = current_pos
            self.last_leg_time = current_time_sec

        except Exception as e:
            self.get_logger().warn(f"[leg_center] TF transform failed: {e}")

    def vector_arrows_callback(self, msg):
        if len(msg.points) > 1:
            self.target_point = msg.points[1]
            x = self.target_point.x
            y = self.target_point.y

            # 嘗試轉換為 rplidar_link 座標系
            local_coords = self.transform_to_local_frame(x, y)

            if local_coords:
                local_x, local_y = local_coords
                self.target_point = Point()
                self.target_point.x = local_x
                self.target_point.y = local_y
                self.target_point.z = 0.0
            else:
                self.get_logger().warn("轉換失敗，local_coords 為 None")
        else:
            self.target_point = None


    def cmd_vel_callback(self, msg):
        # 更新當前速度
        self.current_velocity = msg
    
    def laser_callback(self, msg):
        # 更新雷射資料
        self.laser_data = msg
        self.obstacle_points = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, range_value in enumerate(msg.ranges):
            if math.isinf(range_value) or range_value > 0.1:
                continue  
            angle = angle_min + i * angle_increment
            x = range_value * math.cos(angle)
            y = range_value * math.sin(angle)
            self.obstacle_points.append((x, y))

    def transform_to_local_frame(self, x, y):
        """將全局座標轉換為機器人座標系下的座標"""
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.global_frame  # 通常是 'odom'
            point_stamped.header.stamp =  rclpy.time.Time().to_msg()
            point_stamped.point.x = x
            point_stamped.point.y = y
            point_stamped.point.z = 0.0

            # 執行轉換
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,     #  'rplidar_link'
                self.global_frame,    #  'odom'
                rclpy.time.Time()
            )

            # 轉換點
            local_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)

            return (local_point.point.x, local_point.point.y)

        except TransformException as ex:
            self.get_logger().warning(f"無法轉換座標: {ex}")
            return None

    def check_point_obstacle(self, point, safety_margin=0.25, angle_window=120):
        # 檢查指定點是否與障礙物相撞
        if self.laser_data is None or len(self.laser_data.ranges) == 0:
            return False

        distance = np.hypot(point.x, point.y)
        angle = np.arctan2(point.y, point.x)

        laser_index = int((angle - self.laser_data.angle_min) / self.laser_data.angle_increment)
        if laser_index < 0 or laser_index >= len(self.laser_data.ranges):
            return False

        min_obstacle_distance = float('inf')
        angle_window_size = int(angle_window / 2)
        for i in range(-angle_window_size, angle_window_size + 1):
            index = laser_index + i
            if 0 <= index < len(self.laser_data.ranges):
                laser_distance = self.laser_data.ranges[index]
                if self.laser_data.range_min < laser_distance < self.laser_data.range_max:
                    min_obstacle_distance = min(min_obstacle_distance, laser_distance)

        return min_obstacle_distance < (distance + safety_margin)

    def calculate_goal_score(self, path_points):
        # 計算最後一個點與目標點的距離
        last_point = path_points[-1]
        distance_to_goal = np.hypot(last_point.x - self.target_point.x, last_point.y - self.target_point.y)
        return 1.0 / (1.0 + distance_to_goal)  # 距離越小，分數越高

    def calculate_obstacle_score(self, path_points):
        if not self.obstacle_points:
            return 1.0  
        
            
        max_possible_distance = self.laser_data.range_max  
        
        min_distance_to_obstacle = float('inf')
        for point in path_points:
            for obs_x, obs_y in self.obstacle_points: 
                distance = np.hypot(point.x - obs_x, point.y - obs_y)
                min_distance_to_obstacle = min(min_distance_to_obstacle, distance)
        
        obstacle_score = min(min_distance_to_obstacle / max_possible_distance, 1.0)
    
        return obstacle_score

    
    def calculate_velocity_score(self, v):
        target_speed = getattr(self, 'target_speed', 0.0)
        max_possible_speed = 0.46  # 設最大速度為 1.0 m/s -> 0.46

        # if max_possible_speed == 0:
        #     return 0.0

        score = 1.0 - abs(v - target_speed) / max_possible_speed
        
        return max(0.0, min(1.0, score))

    
    def select_best_path(self, paths):
        # 選擇最佳路徑
        max_score = -float('inf')
        best_path_index = -1
        best_path_points = None
        best_path_velocity = None

        # 計算所有路徑綜合分數
        for idx, (path_points, min_obstacle_distance, v, w) in enumerate(paths):
            goal_score = self.calculate_goal_score(path_points)
            obstacle_score = self.calculate_obstacle_score(path_points)
            velocity_score = self.calculate_velocity_score(v)
            
            total_score = (
                self.obstacle_weight * obstacle_score +
                self.goal_weight * goal_score +
                self.velocity_weight * velocity_score
            )
            # print each score of DWA score
            self.get_logger().info(
                f"Path {idx} | Goal: {goal_score:.3f}, Obstacle: {obstacle_score:.3f}, "
                f"Velocity: {velocity_score:.3f}, Total: {total_score:.3f}"
            )
            if total_score > max_score:
                max_score = total_score
                best_path_index = idx
                best_path_points = path_points
                best_path_velocity = (v, w)
        self.get_logger().info(f"Best Path {best_path_index} | Best Total Score: {max_score:.3f}")

        return best_path_index, best_path_points, best_path_velocity

    def generate_paths(self):
        # 生成多條路徑
        if self.laser_data is None or len(self.laser_data.ranges) == 0:
            self.get_logger().warn("NO LiDAR data")
            return

        delta_v = 0.15
        delta_w = 0.35
        dt = 0.15
        horizon = 30

        v_c = self.current_velocity.linear.x /1.88
        w_c = self.current_velocity.angular.z
        # v_c = 0.5
        # w_c = 0.2
        
        

        marker_array = MarkerArray()
        marker_id = 0
        paths = []
        min_v = v_c-delta_v
        # min_v = max(getattr(self, 'target_speed', 0.0), v_c - delta_v)
        max_v = v_c + delta_v
        for v in np.arange(min_v, max_v , 0.05):
            for w in np.arange(w_c - delta_w, w_c + delta_w , 0.1):
                x, y, theta = 0.0, 0.0, 0.0
                path_points = []

                # 模擬每條路徑
                for _ in range(horizon):
                    x += v * np.cos(theta) * dt
                    y += v * np.sin(theta) * dt
                    theta += w * dt

                    point = Point()
                    point.x = y
                    point.y = -x
                    path_points.append(point)

                has_obstacle = False
                min_obstacle_distance = float('inf')
                for point in path_points:
                    if self.check_point_obstacle(point):
                        has_obstacle = True
                        break
                    distance = np.hypot(point.x, point.y)
                    angle = np.arctan2(point.y, point.x)
                    laser_index = int((angle - self.laser_data.angle_min) / self.laser_data.angle_increment)
                    if 0 <= laser_index < len(self.laser_data.ranges):
                        min_obstacle_distance = min(min_obstacle_distance, self.laser_data.ranges[laser_index])

                if has_obstacle:
                    continue

                paths.append((path_points, min_obstacle_distance, v, w))

        # 選擇最佳路徑
        self.best_path_index, self.best_path_points, self.best_path_velocity = self.select_best_path(paths)
        # if self.best_path_velocity is not None:
            # self.get_logger().info(f"Best Path Velocity: v = {self.best_path_velocity[0]:.3f}, w = {self.best_path_velocity[1]:.3f}")
            
        # 可視化路徑
        for idx, (path_points, obstacle_distance, v, w) in enumerate(paths):
            path_marker = Marker()
            path_marker.header.frame_id = "rplidar_link"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "dwa_paths"
            path_marker.id = marker_id
            marker_id += 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.005
            path_marker.color.a = 1.0

            # 標記最佳路徑
            if idx == self.best_path_index:
                path_marker.color.r = 1.0
                path_marker.color.g = 0.0
                path_marker.color.b = 1.0
            else:
                path_marker.color.r = 0.0
                path_marker.color.g = 1.0
                path_marker.color.b = 0.0

            path_marker.points = path_points
            marker_array.markers.append(path_marker)

        # 刪除多餘的標記
        for i in range(len(marker_array.markers), 100):
            delete_marker = Marker()
            delete_marker.header.frame_id = "rplidar_link"
            delete_marker.ns = "dwa_paths"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)


        # 發佈標記
        self.DWA_path_publisher.publish(marker_array)
        self.send_velcocity_cmd()

    def send_velcocity_cmd(self):
        if self.best_path_velocity is None:
            self.get_logger().warn("No valid path velocity. Skipping velocity command.")
            return

        twist_msg = Twist()
        
        if self.target_point is not None:
            distance = math.hypot(self.target_point.x , self.target_point.y)  

            if distance > 0.20:
                twist_msg.linear.x = self.best_path_velocity[0] *1.88
                twist_msg.angular.z = self.best_path_velocity[1]
            else:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0  

            if twist_msg.linear.x > 0.46:
                twist_msg.linear.x = 0.46

            # print(twist_msg.linear.x)
            # print(twist_msg.angular.z)

            self.cmd_vel_publisher.publish(twist_msg)
        else:
            print("self.target_point is None, skipping velocity calculation.")

 

def main(args=None):
    rclpy.init(args=args)
    node = DWAPathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
