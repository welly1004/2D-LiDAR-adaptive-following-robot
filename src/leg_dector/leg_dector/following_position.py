import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path
import math
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
from nav_msgs.msg import Odometry

class FollowingPositionNode(Node):
    def __init__(self):
        super().__init__('following_position')
        self.subscription = self.create_subscription(
            Marker,
            'leg_center',
            self.leg_center_callback,
            10)
        
        self.publisher = self.create_publisher(Marker, 'following_pos_marker', 10)  
        self.arrow_publisher = self.create_publisher(Marker, '/robot_to_following_position_vector', 10) 
       
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  
            self.lidar_callback,
            10)
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.lidar_data = []  
        self.current_goal = Point()
        self.leg_center_position = None  # Store the leg center position
    
        self.circle_radius_x = 0.5 
        self.circle_radius_y = 0.8  
        self.distance_threshold = 0.35  # 判定距離值

        # 評分權重
        self.w1 = 1.0  # 機器人距離權重
        self.w2 = 1.0  # 障礙物距離權重
        self.w3 = 10.0  # 腿部中心距離權重

    def odom_callback(self, msg):
        self.current_robot_pose = msg.pose.pose
        self.current_goal_pose = self.current_goal

        rounded_robot_x = round(self.current_robot_pose.position.x, 1)
        rounded_robot_y = round(self.current_robot_pose.position.y, 1)
       
        rounded_target_x = round(self.current_goal_pose.x, 1)
        rounded_target_y = round(self.current_goal_pose.y, 1)
        
        # print(rounded_robot_x,rounded_target_x)
        # print(rounded_robot_y,rounded_target_y)

        if (rounded_robot_x == rounded_target_x) and (rounded_robot_y == rounded_target_y):
            print("odom match goal_pose")
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "rplidar_link"  
            goal_pose.header.stamp = self.get_clock().now().to_msg()  

            goal_pose.pose.position.x = 0.0
            goal_pose.pose.position.y = 0.0
            goal_pose.pose.position.z = 0.0

            goal_pose.pose.orientation.w = 1.0
            goal_pose.pose.orientation.x = 0.0
            goal_pose.pose.orientation.y = 0.0
            goal_pose.pose.orientation.z = 0.0

            self.goal_publisher.publish(goal_pose)

    def leg_center_callback(self, msg):
        if msg.points:
            # Store the first leg center point
            self.leg_center_position = msg.points[0]
            for i, point in enumerate(msg.points):
                self.publish_circle_marker(point.x, point.y, i)  
        else:
            self.get_logger().info('No points received in the leg_center message.')
            self.leg_center_position = None

    def publish_goal_pose(self, point):
        self.current_goal = point 
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "rplidar_link"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position = point
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = -0.707  # 90度旋轉
        goal_pose.pose.orientation.w = 0.707  # 對應90度

    def lidar_callback(self, scan_data):
        # 將LiDAR數據轉換為二維點 (x, y)
        self.lidar_data = []
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment

        for i, range_value in enumerate(scan_data.ranges):
            if math.isinf(range_value):
                continue  # 忽略無窮大數據（表示沒有檢測到障礙）
            angle = angle_min + i * angle_increment
            x = range_value * math.cos(angle)
            y = range_value * math.sin(angle)
            self.lidar_data.append((x, y))

    def publish_circle_marker(self, center_x, center_y, marker_id):
        marker = Marker()
        marker.header.frame_id = "rplidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "circle"
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.colors = []
        marker.points = []

        max_score = float('-inf')
        min_score = float('inf')
        highest_score_point = None  

        # 生成圓形點並進行處理
        for circle_point in self.generate_circle_points(center_x, center_y, self.circle_radius_x, self.circle_radius_y):
            distance_to_robot = math.sqrt(circle_point.x**2 + circle_point.y**2)
            distance_to_obstacle = self.closest_obstacle_distance(circle_point.x, circle_point.y)

            # Calculate distance to leg center if available
            leg_center_score = 0
            if self.leg_center_position:
                distance_to_leg = math.sqrt(
                    (circle_point.x - self.leg_center_position.x)**2 + 
                    (circle_point.y - self.leg_center_position.y)**2
                )
                leg_center_score = 1.0 / (distance_to_leg + 1e-5)  # Closer to leg center = higher score

            # Combined weighted score calculation
            score = (
                self.w1 * (1.0 / (distance_to_robot + 1e-5)) - 
                self.w2 * (1.0 / (distance_to_obstacle + 1e-5)) + 
                self.w3 * leg_center_score
            )

            if score > max_score:
                max_score = score
                highest_score_point = circle_point
            if score < min_score:
                min_score = score

            marker.points.append(circle_point)

        # 根據分數設定顏色
        for circle_point in marker.points:
            distance_to_robot = math.sqrt(circle_point.x**2 + circle_point.y**2)
            distance_to_obstacle = self.closest_obstacle_distance(circle_point.x, circle_point.y)
            
            leg_center_score = 0
            if self.leg_center_position:
                distance_to_leg = math.sqrt(
                    (circle_point.x - self.leg_center_position.x)**2 + 
                    (circle_point.y - self.leg_center_position.y)**2
                )
                leg_center_score = 1.0 / (distance_to_leg + 1e-5)

            score = (
                self.w1 * (1.0 / (distance_to_robot + 1e-5)) - 
                self.w2 * (1.0 / (distance_to_obstacle + 1e-5)) + 
                self.w3 * leg_center_score
            )
            normalized_score = (score - min_score) / (max_score - min_score)

            color = ColorRGBA()
            color.r = normalized_score
            color.g = normalized_score
            color.b = normalized_score
            color.a = 1.0

            marker.colors.append(color)

        self.publisher.publish(marker)

        if highest_score_point:
            self.publish_goal_pose(highest_score_point)
            self.publish_vector_arrows("rplidar_link", [(highest_score_point.x, highest_score_point.y)])
    
    def closest_obstacle_distance(self, x, y):
        closest_distance = float('inf')
        for lidar_point in self.lidar_data:
            distance = math.sqrt((lidar_point[0] - x)**2 + (lidar_point[1] - y)**2)
            if distance < closest_distance:
                closest_distance = distance
        return closest_distance

    def publish_vector_arrows(self, frame_id, position_point):
        for i, following_position in enumerate(position_point):
            arrow_marker = Marker()
            arrow_marker.header.frame_id = frame_id
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # 箭頭起點在機器人位置 (0,0)
            arrow_marker.points.append(Point(x=0.0, y=0.0, z=0.0))
            
            # 箭頭終點指向藍色點位置
            arrow_marker.points.append(Point(x=self.current_goal.x, y=self.current_goal.y, z=0.0))
            
            arrow_marker.scale.x = 0.02
            arrow_marker.scale.y = 0.04
            arrow_marker.scale.z = 0.0

            arrow_marker.color.a = 1.0
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 1.0  # 白色箭頭

            # 發佈箭頭
            self.arrow_publisher.publish(arrow_marker)

    def generate_circle_points(self, center_x, center_y, radius_x, radius_y):
        points = []
        num_points = 100  # 圓由 num_points 個點構成
        for i in range(num_points):  
            angle = 2 * math.pi * i / num_points
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            points.append(point)
        return points

def main(args=None):
    rclpy.init(args=args)
    node = FollowingPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()