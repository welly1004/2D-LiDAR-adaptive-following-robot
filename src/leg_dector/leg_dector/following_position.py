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
        self.arrow_pub = self.create_publisher(Marker, 'robot_to_following_position_vector', 10)
        # 添加分數標籤的發布者
        self.score_text_pub = self.create_publisher(MarkerArray, 'score_text_markers', 10)

        self.long_axis_arrow_pub = self.create_publisher(Marker, 'long_axis_vector', 10)
        self.best_point_vector_pub = self.create_publisher(Marker, 'best_point_vector', 10)
        self.next_leg_pub = self.create_publisher(Marker, 'next_leg_marker', 10)
        
        # 初始化TF2緩衝區和監聽器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.circle_radius_x = 0.8
        self.circle_radius_y = 0.5  
        self.leg_positions = deque(maxlen=5)      # 相對座標下的腿部位置
        self.global_leg_positions = deque(maxlen=5)  # 全局座標下的腿部位置
        self.current_theta = 0.0
        self.global_theta = 0.0  # 全局座標系中的方向角
        self.movement_threshold = 0.30

        self.alpha = 0.45 #robot to following position 0.15
        self.beta =  0.48 #obstacle to following position
        self.gamma = 0.22 #leg to following position 0.37
        self.max_obs_dist = 0.35
        
        self.robot_pos = (0.0, 0.0)
        self.obstacles = []
        
        # 設置坐標系（使用odom作為全局參考系）
        self.global_frame = "odom"
        self.robot_frame = "rplidar_link"
        
        # 是否顯示所有點的分數
        self.show_all_scores = True
        # 顯示總分或分項分數
        self.show_detailed_scores = False
        self.last_marker_count = 0

        self.prev_leg_time = None
        self.prev_leg_pos = None

        # 記錄下一個橢圓中心點的全局座標
        self.next_leg_global = None
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

    def transform_from_global_to_robot_frame(self, x, y):
        """將全局座標轉換為機器人座標系下的坐標"""
        try:
            # 創建帶時間戳的點
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.global_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = x
            point_stamped.point.y = y
            point_stamped.point.z = 0.0
            
            # 執行轉換
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.global_frame,
                rclpy.time.Time())
                
            # 轉換點
            local_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            return (local_point.point.x, local_point.point.y)
        except TransformException as ex:
            self.get_logger().warning(f"無法轉換座標: {ex}")
            return None

    def calculate_detailed_score(self, x, y):
        """計算點 (x, y) 的詳細分數，所有分數都使用正值表示更優的選擇"""
        # epsilon = 0.1  # 防止除以 0
        
        # --- 機器人分數（離機器人越近分數越高）---
        d_robot = math.sqrt((x - self.robot_pos[0])**2 + (y - self.robot_pos[1])**2)
        d_robot = abs(d_robot)
        # 將距離轉換為 0-1 範圍的分數，越近分數越高
        max_robot_dist = 2.0  # 設定一個最大距離閾值
        # robot_score = max(0, 1.0 - d_robot / max_robot_dist)
        robot_score = 1.0 / (1.0 + d_robot)
        robot_score = self.alpha * robot_score  # 應用權重
        
        # --- 障礙物分數（離障礙物越遠分數越高）---
        if not self.obstacles:
            obstacle_score = 1.0 # 如果沒有障礙物，給予最高分
        else:
            
            # 將障礙物資料轉換為 NumPy 陣列
            obstacles = np.array(self.obstacles)

            # 計算與所有障礙物的距離
            dists = np.linalg.norm(obstacles - np.array([x, y]), axis=1)

            # 取得最小距離
            min_obstacle_dist = np.min(dists)
            min_obstacle_dist = abs(min_obstacle_dist)

            # 將距離轉換為 0-1 範圍的分數，越遠分數越高
            obstacle_score = min(1.0, min_obstacle_dist / self.max_obs_dist)
            obstacle_score = self.beta * obstacle_score
        
        # --- 腳分數（離腳越近分數越高）---
        if self.next_leg_global:
            # 將 next_leg_global (odom) 轉到 robot frame
            local_next_leg = self.transform_from_global_to_robot_frame(
                self.next_leg_global[0], self.next_leg_global[1])
            
            if local_next_leg:
                leg_x, leg_y = local_next_leg
                d_leg = math.sqrt((x - leg_x)**2 + (y - leg_y)**2)
                d_leg = abs(d_leg)
                # print(d_leg)
                # max_leg_dist = 0.85
                leg_score = max((0.8 - d_leg )/ 0.3,0)
                # print(leg_score)
                leg_score = self.gamma * leg_score
            else:
                leg_score = 0.0
        else:
            leg_score = 0.0
        
        # --- 總分 (所有分數都是正值，越高越好) ---
        total_score = robot_score + obstacle_score + leg_score
        
        return robot_score, obstacle_score, leg_score, total_score

    def transform_angle_to_robot_frame(self, global_angle):
        """將角度從全局座標系（odom）轉換到機器人座標系"""
        try:
            # 獲取從odom到機器人座標系的變換
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.global_frame,
                rclpy.time.Time())
                
            # 從變換中提取機器人相對於odom的旋轉角度
            q = transform.transform.rotation
            robot_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                
            # 將全局角度轉換為機器人座標系
            robot_relative_theta = global_angle - robot_theta
            
            # 確保角度在[-pi, pi]範圍內
            while robot_relative_theta > math.pi:
                robot_relative_theta -= 2 * math.pi
            while robot_relative_theta < -math.pi:
                robot_relative_theta += 2 * math.pi
                
            # self.get_logger().info(f"機器人座標系下的移動方向: {robot_relative_theta:.2f} 弧度")
            return robot_relative_theta
            
        except TransformException as ex:
            self.get_logger().warning(f"無法獲取機器人變換: {ex}")
            return self.current_theta  # 變換失敗時回退到當前角度

    def publish_arrow(self, start_point, best_point):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rclpy.time.Time().to_msg()  # 避免TF錯誤
        marker.ns = "best_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.points = []

        # Z軸高度
        z_height = 0.1

        global_start = self.transform_to_global_frame(start_point[0], start_point[1])
        if global_start is None:
            return

        start = Point()
        start.x = global_start[0]
        start.y = global_start[1]
        start.z = z_height
        marker.points.append(start)

        end = Point()
        end.x = best_point[0]
        end.y = best_point[1]
        end.z = z_height
        marker.points.append(end)

        marker.scale.x = 0.02  # 桿寬
        marker.scale.y = 0.04  # 頭寬
        marker.scale.z = 0.2   # 頭長（要是正的）

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.arrow_pub.publish(marker)



    def generate_ellipse_points_with_scores_in_global_frame(self, center_x, center_y, theta):
        """在全局座標系中生成橢圓形的點和分數"""
        points_and_scores = []
        num_points = 50
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # 先將中心點轉換到全局座標系
        global_center = self.transform_to_global_frame(center_x, center_y)
        if global_center is None:
            return []
        
        center_x_global, center_y_global = global_center
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = self.circle_radius_x * math.cos(angle)
            y = self.circle_radius_y * math.sin(angle)
            
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            
            final_x = center_x_global + x_rot
            final_y = center_y_global + y_rot
            
            point = Point()
            point.x = final_x
            point.y = final_y
            point.z = 0.0
            
            # 獲取詳細分數
            # 如果分數計算需要在機器人座標系中，先轉換回機器人座標系
            local_point = self.transform_from_global_to_robot_frame(final_x, final_y)
            if local_point:
                scores = self.calculate_detailed_score(local_point[0], local_point[1])
            else:
                scores = (0, 0, 0, 0)  # 默認分數
            
            points_and_scores.append((point, scores))
            
        return points_and_scores

    def publish_ellipse(self, center_x, center_y, theta):
        marker = Marker()
        marker.header.frame_id = self.global_frame  # 使用全局參考系
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = "ellipse"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # 使用全局座標系和全局方向生成橢圓點
        points_and_scores = self.generate_ellipse_points_with_scores_in_global_frame(center_x, center_y, self.global_theta)
        
        if not points_and_scores:
            self.get_logger().warning("無法生成橢圓點")
            return
        
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
        
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        
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
        text_marker_array = MarkerArray()

        for i, (point, score_tuple) in enumerate(points_and_scores):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = "score_labels"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # 計算偏移量，避免文字和點重疊
            dx = point.x - center_x  
            dy = point.y - center_y  
            norm = math.hypot(dx, dy)
            
            # 計算偏移向量（單位方向 * 偏移距離）
            offset_dist = 0.0 # 偏移的距離，可以調整
            offset_x = (dx / norm) * offset_dist if norm != 0 else 0.0
            offset_y = (dy / norm) * offset_dist if norm != 0 else 0.0

            # 將文字稍微抬高並偏移
            marker.pose.position.x = point.x + offset_x
            marker.pose.position.y = point.y + offset_y
            marker.pose.position.z = 0.1 # 這裡是讓文字往上浮一些，避免與點重疊

            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.03  # 控制文字大小
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.text = f"{score_tuple[3]:.2f}"

            # 將 marker 加入到 MarkerArray 中
            text_marker_array.markers.append(marker)

        # 發布 MarkerArray
        self.score_text_pub.publish(text_marker_array)

        # 刪除多出來的舊 markers
        for j in range(len(points_and_scores), self.last_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = self.global_frame
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "score_labels"
            delete_marker.id = j
            delete_marker.action = Marker.DELETE
            text_marker_array.markers.append(delete_marker)

        # 更新記錄
        self.last_marker_count = len(points_and_scores)

        # 發布
        self.score_text_pub.publish(text_marker_array)

        if best_point:
            # 圓心的全局座標
            center_global = self.transform_to_global_frame(center_x, center_y)
            if center_global is not None:
                self.publish_long_axis_vector(center_global, self.global_theta)
                self.publish_best_point_vector(center_global, best_point)

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

    def calculate_movement_direction(self):
        """計算全局座標系（odom）中的移動方向"""
        if len(self.global_leg_positions) < 2:
            return None
            
        # 獲取odom座標系中最早和最新的腿部位置
        x_old, y_old = self.global_leg_positions[0]
        x_new, y_new = self.global_leg_positions[-1]
        
        # 計算實際移動距離
        distance = math.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
        
        # 如果距離小於閾值，認為沒有實質性移動
        if distance < self.movement_threshold:
            # self.get_logger().info(f"移動距離 {distance:.2f} 小於閾值 {self.movement_threshold}，不更新方向")
            return None
        
        # 計算odom座標系中的移動向量和方向
        vx, vy = x_new - x_old, y_new - y_old
        
        # 計算odom座標系中的方向
        global_theta = math.atan2(vy, vx)
        # self.get_logger().info(f"odom座標系下的移動方向: {global_theta:.2f} 弧度")
        
        # 返回全局方向，不轉換到機器人座標系
        return global_theta
    def leg_center_callback(self, msg):
        if not msg.points:
            return

        local_leg_pos = (msg.points[0].x, msg.points[0].y)
        self.leg_positions.append(local_leg_pos)

        global_leg_pos = self.transform_to_global_frame(local_leg_pos[0], local_leg_pos[1])
        if global_leg_pos is None:
            return

        self.global_leg_positions.append(global_leg_pos)

        now = self.get_clock().now()
        prediction_dt = 0.45 # 預測未來 1 秒

        if self.prev_leg_time and self.prev_leg_pos:
            dt = (now - self.prev_leg_time).nanoseconds / 1e9
            if dt > 0:
                dx = global_leg_pos[0] - self.prev_leg_pos[0]
                dy = global_leg_pos[1] - self.prev_leg_pos[1]
                vx = dx / dt
                vy = dy / dt
                speed = math.hypot(vx, vy)
                distance = math.hypot(dx, dy)

                if distance < 0.05 or speed < 0.05:
                    next_leg_x, next_leg_y = global_leg_pos
                else:
                    next_leg_x = global_leg_pos[0] + vx * prediction_dt
                    next_leg_y = global_leg_pos[1] + vy * prediction_dt
            else:
                next_leg_x, next_leg_y = global_leg_pos
        else:
            next_leg_x, next_leg_y = global_leg_pos

        # 更新記錄
        self.prev_leg_time = now
        self.prev_leg_pos = global_leg_pos

        # 儲存與發佈
        self.next_leg_global = (next_leg_x, next_leg_y)
        self.publish_next_leg_marker(next_leg_x, next_leg_y)

        # 計算方向角
        new_global_theta = self.calculate_movement_direction()
        if new_global_theta is not None:
            self.global_theta = new_global_theta

        # 發佈橢圓
        next_leg_local = self.transform_from_global_to_robot_frame(next_leg_x, next_leg_y)
        if next_leg_local:
            self.publish_ellipse(next_leg_local[0], next_leg_local[1], self.global_theta)


    # def leg_center_callback(self, msg):
    #     if msg.points:
    #         # 保存機器人相對座標系下的腿部位置
    #         local_leg_pos = (msg.points[0].x, msg.points[0].y)
    #         self.leg_positions.append(local_leg_pos)
            
    #         # 轉換為odom座標系並保存
    #         global_leg_pos = self.transform_to_global_frame(local_leg_pos[0], local_leg_pos[1])
    #         if global_leg_pos is not None:
    #             self.global_leg_positions.append(global_leg_pos)
                
    #             # 計算全局座標系中的移動方向
    #             new_global_theta = self.calculate_movement_direction()
    #             if new_global_theta is not None:
    #                 self.global_theta = new_global_theta  # 存儲全局方向
    #                 self.get_logger().info(f"更新全局移動方向: {self.global_theta:.2f} 弧度")
                
    #             # 使用原始相對座標和全局方向發布可視化標記
    #             self.publish_ellipse(local_leg_pos[0], local_leg_pos[1], self.global_theta)
    def publish_long_axis_vector(self, center, theta):
        # 粉紅色箭頭，圓心指向橢圓長軸方向
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "long_axis_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x = center[0]
        start.y = center[1]
        start.z = 0.0

        # 橢圓長軸方向向量長度設為0.5m，可調整
        length = 0.5
        end = Point()
        end.x = center[0] + length * math.cos(theta)
        end.y = center[1] + length * math.sin(theta)
        end.z = 0.0

        marker.points = [start, end]

        marker.scale.x = 0.03  # 箭頭桿寬
        marker.scale.y = 0.06  # 箭頭頭寬
        marker.scale.z = 0.1   # 箭頭頭長

        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.7
        marker.color.a = 1.0

        self.long_axis_arrow_pub.publish(marker)


    def publish_best_point_vector(self, center, best_point):
        # 紫色箭頭，圓心指向最佳點
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_point_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x = center[0]
        start.y = center[1]
        start.z = 0.0

        end = Point()
        end.x = best_point[0]
        end.y = best_point[1]
        end.z = 0.0

        marker.points = [start, end]

        marker.scale.x = 0.03
        marker.scale.y = 0.06
        marker.scale.z = 0.1

        marker.color.r = 0.5
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.color.a = 1.0

        self.best_point_vector_pub.publish(marker)
    
    def publish_next_leg_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.next_leg_pub.publish(marker)
def main(args=None):
    rclpy.init(args=args)
    node = FollowingPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
