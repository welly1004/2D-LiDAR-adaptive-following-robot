import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
import math
from collections import deque

class FollowingPositionNode(Node):
    def __init__(self):
        super().__init__('following_position')
        self.subscription = self.create_subscription(
            Marker, 'leg_center', self.leg_center_callback, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        
        # 添加箭頭的發布者
        self.publisher = self.create_publisher(Marker, 'following_pos_marker', 10)
        self.arrow_publisher = self.create_publisher(Marker, 'robot_to_following_position_vector', 10)
        
        self.circle_radius_x = 0.8
        self.circle_radius_y = 0.5  
        self.leg_positions = deque(maxlen=5)
        self.current_theta = 0.0
        self.movement_threshold = 0.25
        
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 5.0
        self.max_obs_dist = 2.0
        
        self.robot_pos = (0.0, 0.0)
        self.obstacles = []

    def calculate_detailed_score(self, x, y):
        # 計算各個分項分數
        d_robot = math.sqrt((x - self.robot_pos[0])**2 + (y - self.robot_pos[1])**2)
        robot_score = -self.alpha * d_robot
        
        # 修改障礙物分數計算
        d_obstacle = float('inf')
        for obs in self.obstacles:
            dist = math.sqrt((x - obs[0])**2 + (y - obs[1])**2)
            d_obstacle = min(d_obstacle, dist)

        # 使用反比關係計算分數：距離越小，分數越低（更負）
        # 添加一個小常數避免除以零的問題
        epsilon = 0.05  # 小常數，避免分母為零
        obstacle_score = -self.beta / (d_obstacle + epsilon)
        
        leg_score = 0
        if self.leg_positions:
            leg_x, leg_y = self.leg_positions[-1]
            d_leg = math.sqrt((x - leg_x)**2 + (y - leg_y)**2)
            leg_score = -self.gamma * d_leg
        
        total_score = robot_score + obstacle_score + leg_score
        return robot_score, obstacle_score, leg_score, total_score

    def publish_arrow(self, start_point, best_point):
        marker = Marker()
        marker.header.frame_id = "rplidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 設置箭頭起點和終點
        marker.points = []
        start = Point()
        start.x = start_point[0]
        start.y = start_point[1]
        start.z = 0.0
        marker.points.append(start)
        
        end = Point()
        end.x = best_point[0]
        end.y = best_point[1]
        end.z = 0.0
        marker.points.append(end)
        
        # 設置箭頭的大小
        marker.scale.x = 0.02  # 箭桿寬度
        marker.scale.y = 0.04   # 箭頭寬度
        marker.scale.z = 0.0   # 箭頭長度
        
        # 設置箭頭顏色（紅色）
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        self.arrow_publisher.publish(marker)

    def publish_ellipse(self, center_x, center_y, theta):
        marker = Marker()
        marker.header.frame_id = "rplidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ellipse"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        points_and_scores = self.generate_ellipse_points_with_scores(center_x, center_y, theta)
        
        # 找出最高分數的點
        best_point = None
        best_score = float('-inf')
        scores = []
        
        # 輸出所有點的分數
        self.get_logger().info("\n=== Points Scores ===")
        
        for i, (point, score_tuple) in enumerate(points_and_scores):
            robot_score, obstacle_score, leg_score, total_score = score_tuple
            scores.append(total_score)
            
            # 輸出每個點的詳細分數
            self.get_logger().info(f"Point {i}: Robot Score: {robot_score:.2f}, "
                                 f"Obstacle Score: {obstacle_score:.2f}, "
                                 f"Leg Score: {leg_score:.2f}, "
                                 f"Total Score: {total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_point = (point.x, point.y)
        
        min_score = min(scores)
        max_score = max(scores)
        
        # 發布指向最佳點的箭頭
        if best_point:
            self.publish_arrow((0.0, 0.0), best_point)
        
        marker.points = []
        marker.colors = []
        for point, score_tuple in points_and_scores:
            marker.points.append(point)
            
            color = ColorRGBA()
            r, g, b = self.get_color_from_score(score_tuple[3], min_score, max_score)
            color.r = float(r)
            color.g = float(g)
            color.b = float(b)
            color.a = 1.0
            marker.colors.append(color)
        
        self.publisher.publish(marker)

    def generate_ellipse_points_with_scores(self, center_x, center_y, theta):
        points_and_scores = []
        num_points = 100
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = self.circle_radius_x * math.cos(angle)
            y = self.circle_radius_y * math.sin(angle)
            
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            
            final_x = center_x + x_rot
            final_y = center_y + y_rot
            
            point = Point()
            point.x = final_x
            point.y = final_y
            point.z = 0.0
            
            # 獲取詳細分數
            scores = self.calculate_detailed_score(final_x, final_y)
            
            points_and_scores.append((point, scores))
            
        return points_and_scores

    
    def scan_callback(self, msg):
        self.obstacles = []
        angle = msg.angle_min
        for distance in msg.ranges:
            if msg.range_min <= distance <= msg.range_max:
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                self.obstacles.append((x, y))
            angle += msg.angle_increment

    def get_color_from_score(self, score, min_score, max_score):
        if max_score == min_score:
            normalized = 0.5
        else:
            normalized = (score - min_score) / (max_score - min_score)
        return normalized, normalized, normalized

    def leg_center_callback(self, msg):
        if msg.points:
            self.leg_positions.append((msg.points[0].x, msg.points[0].y))
            new_theta = self.calculate_movement_direction()
            if new_theta is not None:
                self.current_theta = new_theta
            self.publish_ellipse(msg.points[0].x, msg.points[0].y, self.current_theta)

    def calculate_movement_direction(self):
        if len(self.leg_positions) < 2:
            return None
        x_old, y_old = self.leg_positions[0]
        x_new, y_new = self.leg_positions[-1]
        distance = math.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
        if distance < self.movement_threshold:
            return None
        vx, vy = x_new - x_old, y_new - y_old
        return math.atan2(vy, vx)

def main(args=None):
    rclpy.init(args=args)
    node = FollowingPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
