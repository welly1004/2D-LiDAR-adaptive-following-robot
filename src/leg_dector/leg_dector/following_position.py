import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
import math
from collections import deque
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
import numpy as np

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
        # 添加分數標籤的發布者
        self.score_publisher = self.create_publisher(MarkerArray, 'point_scores', 10)
        
        # 初始化TF2緩衝區和監聽器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.circle_radius_x = 0.8
        self.circle_radius_y = 0.5  
        self.leg_positions = deque(maxlen=5)      # 相對座標下的腿部位置
        self.global_leg_positions = deque(maxlen=5)  # 全局座標下的腿部位置
        self.current_theta = 0.0
        self.movement_threshold = 0.2
        
        self.alpha = 0.0 #robot to following position
        self.beta = 1.0  #obstacle to following position
        self.gamma = 0.0 #leg to following position
        self.max_obs_dist = 2.0
        
        self.robot_pos = (0.0, 0.0)
        self.obstacles = []
        
        # 設置坐標系（使用odom作為全局參考系）
        self.global_frame = "odom"
        self.robot_frame = "rplidar_link"
        
        # 是否顯示所有點的分數
        self.show_all_scores = True
        # 顯示總分或分項分數
        self.show_detailed_scores = False

    def transform_to_global_frame(self, x, y):
        """將相對座標轉換為odom坐標系下的坐標"""
        try:
            # 創建帶時間戳的點
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.robot_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = x
            point_stamped.point.y = y
            point_stamped.point.z = 0.0
            
            # 執行轉換
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time())
                
            # 轉換點
            global_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            return (global_point.point.x, global_point.point.y)
        except TransformException as ex:
            self.get_logger().warning(f"無法轉換座標: {ex}")
            return None

    def calculate_detailed_score(self, x, y):
        """計算點 (x, y) 的詳細分數，所有分數都使用正值表示更優的選擇"""
        # epsilon = 0.1  # 防止除以 0
        
        # --- 機器人分數（離機器人越近分數越高）---
        d_robot = math.sqrt((x - self.robot_pos[0])**2 + (y - self.robot_pos[1])**2)
        # 將距離轉換為 0-1 範圍的分數，越近分數越高
        max_robot_dist = 2.0  # 設定一個最大距離閾值
        robot_score = max(0, 1.0 - d_robot / max_robot_dist)
        robot_score = self.alpha * robot_score  # 應用權重
        
        # --- 障礙物分數（離障礙物越遠分數越高）---
        if not self.obstacles:
            obstacle_score = self.beta  # 如果沒有障礙物，給予最高分
        else:
            
            # 將障礙物資料轉換為 NumPy 陣列
            obstacles = np.array(self.obstacles)

            # 計算與所有障礙物的距離
            dists = np.linalg.norm(obstacles - np.array([x, y]), axis=1)

            # 取得最小距離
            min_obstacle_dist = np.min(dists)

            # 將距離轉換為 0-1 範圍的分數，越遠分數越高
            obstacle_score = min(1.0, min_obstacle_dist / self.max_obs_dist)
        
        # --- 腳分數（離腳越近分數越高）---
        if self.leg_positions:
            leg_x, leg_y = self.leg_positions[-1]
            d_leg = math.sqrt((x - leg_x)**2 + (y - leg_y)**2)
            # 將距離轉換為 0-1 範圍的分數，越近分數越高
            max_leg_dist = 1.5  # 設定一個最大距離閾值
            leg_score = max(0, 1.0 - d_leg / max_leg_dist)
            leg_score = self.gamma * leg_score  # 應用權重
        else:
            leg_score = 0.0
        
        # --- 總分 (所有分數都是正值，越高越好) ---
        total_score = robot_score + obstacle_score + leg_score
        
        return robot_score, obstacle_score, leg_score, total_score

    def publish_arrow(self, start_point, best_point):
        marker = Marker()
        marker.header.frame_id = self.robot_frame
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
        
        # 設置箭頭顏色（白色）
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        self.arrow_publisher.publish(marker)

    def publish_score_labels(self, points_and_scores):
        """發布分數標籤作為文本標記"""
        marker_array = MarkerArray()
        
        for i, (point, score_tuple) in enumerate(points_and_scores):
            robot_score, obstacle_score, leg_score, total_score = score_tuple
            
            # 只發布部分點的分數標籤（每隔幾個點發布一個）或特別高分的點
            if not self.show_all_scores and i % 5 != 0 and total_score < 0.8 * max([s[3] for _, s in points_and_scores]):
                continue
                
            marker = Marker()
            marker.header.frame_id = self.robot_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "score_labels"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # 計算標籤的偏移位置
        # 根據點在橢圓上的位置，將標籤向外偏移
            vector_to_center_x = point.x - self.leg_positions[-1][0] if self.leg_positions else point.x
            vector_to_center_y = point.y - self.leg_positions[-1][1] if self.leg_positions else point.y
            
            # 計算向量長度
            vector_length = math.sqrt(vector_to_center_x**2 + vector_to_center_y**2)
            
            # 標準化向量並設置偏移距離
            offset_distance = 0.15  # 調整此值可以改變標籤與點的距離
            
            if vector_length > 0.01:  # 避免除以零
                # 將標籤沿著從中心向外的方向偏移
                offset_x = (vector_to_center_x / vector_length) * offset_distance
                offset_y = (vector_to_center_y / vector_length) * offset_distance
            else:
                # 如果點非常接近中心，使用默認偏移
                offset_x = offset_distance
                offset_y = 0
            
            # 設置文本位置（向外偏移以避免與點重疊）
            marker.pose.position.x = point.x + offset_x
            marker.pose.position.y = point.y + offset_y
            marker.pose.position.z = 0.1  # 稍微抬高文本
            marker.pose.orientation.w = 1.0
            
            # 設置文本大小
            marker.scale.z = 0.05  # 文本高度
            
            # 根據分數設置顏色（使用與點相同的顏色映射）
            r, g, b = self.get_color_from_score(total_score, 
                                            min([s[3] for _, s in points_and_scores]),
                                            max([s[3] for _, s in points_and_scores]))
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 1.0
            
            # 設置文本內容
            if self.show_detailed_scores:
                marker.text = f"R:{robot_score:.1f} O:{obstacle_score:.1f} L:{leg_score:.1f} T:{total_score:.1f}"
            else:
                marker.text = f"{total_score:.2f}"
                
            marker_array.markers.append(marker)
            
        self.score_publisher.publish(marker_array)

    def publish_ellipse(self, center_x, center_y, theta):
        marker = Marker()
        marker.header.frame_id = self.robot_frame
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
        
        # 發布分數標籤
        self.publish_score_labels(points_and_scores)

    def generate_ellipse_points_with_scores(self, center_x, center_y, theta):
        points_and_scores = []
        num_points = 50
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
            # 保存相對座標系下的腿部位置
            local_leg_pos = (msg.points[0].x, msg.points[0].y)
            self.leg_positions.append(local_leg_pos)
            
            # 轉換為odom座標系並保存
            global_leg_pos = self.transform_to_global_frame(local_leg_pos[0], local_leg_pos[1])
            if global_leg_pos is not None:
                self.global_leg_positions.append(global_leg_pos)
                
                # 使用odom座標系下的位置計算移動方向
                new_theta = self.calculate_movement_direction()
                if new_theta is not None:
                    self.current_theta = new_theta
                    self.get_logger().info(f"更新移動方向: {self.current_theta:.2f} 弧度")
                    
            # 使用原始相對座標發布可視化標記
            self.publish_ellipse(local_leg_pos[0], local_leg_pos[1], self.current_theta)

    def calculate_movement_direction(self):
        """基於odom座標系計算移動方向"""
        if len(self.global_leg_positions) < 2:
            return None
            
        # 取最早和最新的odom座標系下的腳部位置
        x_old, y_old = self.global_leg_positions[0]
        x_new, y_new = self.global_leg_positions[-1]
        
        # 計算實際移動距離
        distance = math.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
        
        # 如果距離小於閾值，認為腳部沒有實質性移動
        if distance < self.movement_threshold:
            self.get_logger().info(f"移動距離 {distance:.2f} 小於閾值 {self.movement_threshold}，不更新方向")
            return None
        
        # 計算移動向量和方向
        vx, vy = x_new - x_old, y_new - y_old
        
        # 計算在odom座標系下的方向
        global_theta = math.atan2(vy, vx)
        self.get_logger().info(f"odom座標系下的移動方向: {global_theta:.2f} 弧度")
        
        # 獲取機器人當前的變換以將odom座標系下的方向轉換回機器人坐標系
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.global_frame,
                rclpy.time.Time())
                
            # 從變換中提取機器人的旋轉角度
            q = transform.transform.rotation
            robot_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                
            # 將odom座標系下的方向轉換為相對於機器人的方向
            relative_theta = global_theta - robot_theta
            
            # 確保角度在[-pi, pi]範圍內
            while relative_theta > math.pi:
                relative_theta -= 2 * math.pi
            while relative_theta < -math.pi:
                relative_theta += 2 * math.pi
                
            self.get_logger().info(f"機器人坐標系下的移動方向: {relative_theta:.2f} 弧度")
            return relative_theta
            
        except TransformException as ex:
            self.get_logger().warning(f"無法獲取機器人變換: {ex}")
            return None
    
def main(args=None):
    rclpy.init(args=args)
    node = FollowingPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
