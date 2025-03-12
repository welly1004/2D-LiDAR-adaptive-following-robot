import rclpy, math, time
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy



class DWAPathPlanner(Node):
    def __init__(self):
        super().__init__('dwa_path_planner')

        # 初始化參數
        self.obstacle_weight = 1.0  # 障礙物權重
        self.goal_weight = 5.0  # 目標點權重
        self.velocity_weight = 0.5  # 速度權重

        # 發佈路徑
        self.DWA_path_publisher = self.create_publisher(MarkerArray, '/dwa_paths', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        )
        self.angle_subscriber = self.create_subscription(
            Marker,
            # '/robot_to_leg_vector',
            '/robot_to_following_position_vector',
            self.vector_arrows_callback,
            10)
        # 訂閱雷射資料
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        self.laser_data = None
        self.current_velocity = Twist()
        self.timer = self.create_timer(0.1, self.generate_paths)

        self.target_point = Point(x=0.0, y=0.0, z=0.0) 

        self.best_path_points = None 
        self.best_path_index = -1
        self.best_path_velocity = None
   

    def vector_arrows_callback(self,msg):
        if len(msg.points) > 1:
            self.target_point = msg.points[1]
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



    def check_point_obstacle(self, point, safety_margin=0.2, angle_window=15):
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
        max_possible_speed = 1.0  
        return v / max_possible_speed
    
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
            # self.get_logger().info(
            #     f"Path {idx} | Goal: {goal_score:.3f}, Obstacle: {obstacle_score:.3f}, "
            #     f"Velocity: {velocity_score:.3f}, Total: {total_score:.3f}"
            # )
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

        delta_v = 0.1
        delta_w = 0.5
        dt = 0.1
        horizon = 20

        v_c = self.current_velocity.linear.x
        w_c = self.current_velocity.angular.z
        # v_c = 0.5
        # w_c = 0.2
        
        

        marker_array = MarkerArray()
        marker_id = 0
        paths = []

        for v in np.arange(v_c - delta_v, v_c + delta_v , 0.05):
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
        if self.best_path_velocity is not None:
            self.get_logger().info(f"Best Path Velocity: v = {self.best_path_velocity[0]:.3f}, w = {self.best_path_velocity[1]:.3f}")
            
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
        twist_msg= Twist()
        # twist_msg.linear.x =   self.best_path_velocity[0]  # Set linear velocity
        if self.target_point is not None:
            if self.best_path_velocity[0] < 0.15:
                twist_msg.linear.x =   self.best_path_velocity[0]
                twist_msg.angular.z =  0.0
                # twist_msg.angular.z =  0.0 # Set angular velocity left is + right is -        
            else:
                twist_msg.linear.x =   self.best_path_velocity[0]
                twist_msg.angular.z =  self.best_path_velocity[1] # Set angular velocity left is + right is -
            # twist_msg.angular.z =  self.best_path_velocity[1]
            print(twist_msg.linear.x)
            print(twist_msg.angular.z)
            # Publish Twist message
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
