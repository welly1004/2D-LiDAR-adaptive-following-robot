import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.dates as mdates
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from geometry_msgs.msg import Point, Quaternion

import math
import time




def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = quaternion
    n = x*x + y*y + z*z + w*w
    if n < np.finfo(float).eps:
        return np.identity(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array([
        [1.0 - (yy + zz),     xy - wz,         xz + wy],
        [    xy + wz,     1.0 - (xx + zz),     yz - wx],
        [    xz - wy,         yz + wx,     1.0 - (xx + yy)]
    ])

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion.
    Args:
        roll: rotation around x-axis in radians
        pitch: rotation around y-axis in radians
        yaw: rotation around z-axis in radians
    Returns:
        Quaternion as [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]
class LegDetectionNode(Node):
    def __init__(self):
        super().__init__('leg_detection')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.marker_publisher = self.create_publisher(Marker, '/leg_center', 10) 
        self.arrow_publisher = self.create_publisher(Marker, '/robot_to_leg_vector', 10) 
        self.front_arrow_publisher = self.create_publisher(Marker, '/robot_front_vector', 10) 
        self.Mr_Chou_publisher = self.create_publisher(Marker, '/Mr_Chou', 10) 
        self.distance_publisher = self.create_publisher(Marker, '/robot_to_leg_distance', 10) 
        self.cluster_marker_publisher = self.create_publisher(Marker, '/leg_marker', 10)
        self.go_back_publisher = self.create_publisher(Bool, '/go_back', 10)
        # self.start_end_publisher = self.create_publisher(Bool, '/start_end', 10)
        self.start_position_publisher = self.create_publisher(PoseStamped, '/start_position', 10)
        self.shutdown_distance_threshold = 0.2  # 關閉節點的距離閾值


        self.robot_trajectory_publisher = self.create_publisher(Path, '/robot_trajectory', 10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.robot_trajectory = Path()
        self.robot_trajectory.header.frame_id = 'odom'  # Assuming odom as the global frame
        
        

        self.path_pub = self.create_publisher(Path, '/target_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.timer = self.create_timer(0.1, self.publish_path)  # 每100毫秒發布一次
        
        self.start_pose_published = False
        # temp = Bool()
        # temp.data = False
        # self.go_back_publisher.publish(temp)
        
        # Initialize for plotting
        # self.distances = []
        # self.timestamps = []
        # plt.style.use('seaborn-darkgrid')
        # self.fig, self.ax = plt.subplots()
        # # self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000)
        # plt.xlabel('Time')
        # plt.ylabel('Distance (m)')
        # plt.title('Real-time Distance to Leg')
        # plt.ion()  # Enable interactive mode
        self.last_target_position = None
        self.last_stable_time = time.time()
        self.position_tolerance = 0.05  # 小於這個值就當作沒動
        self.stability_duration = 200.0   # 5 秒沒動就 shutdown
    
        self.start_pose_published = False
    def odom_callback(self, msg):
        self.current_robot_pose = msg.pose.pose
        # print("Odometry received: ", self.current_robot_pose)
         # 如果已檢測到目標，記錄當前的位姿並發布 START 位置
        if hasattr(self, 'target_detected') and self.target_detected:
            if not self.start_pose_published:  # 如果還沒有發布過
                # 創建一個新的 PoseStamped 訊息來存儲當前位姿
                start_pose = PoseStamped()
                start_pose.header = msg.header
                start_pose.pose = self.current_robot_pose
                self.start_position_publisher.publish(start_pose)
                # self.get_logger().info(f"{start_pose}")
                # 設置標誌為 True，表示已經發布過一次
                self.start_pose_published = True
            
        robot_pose = PoseStamped()
        robot_pose.header = msg.header
        robot_pose.pose = self.current_robot_pose
        self.robot_trajectory.poses.append(robot_pose)
        
        self.robot_trajectory_publisher.publish(self.robot_trajectory)

    def publish_path(self):
        # if hasattr(self, 'target_position'):
        #     print("target_position ok")
        # if hasattr(self, 'current_robot_pose'):
        #     print("current_robot_pose ok")
        # 創建當前目標的位姿
        if hasattr(self, 'target_position') and hasattr(self, 'current_robot_pose'):
            x,y=self.target_position
            # print(self.target_position)
        else:
            
            return
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        # pose.pose=self.current_robot_pose
        rotation_matrix=quaternion_to_rotation_matrix( [
    self.current_robot_pose.orientation.x,
    self.current_robot_pose.orientation.y,
    self.current_robot_pose.orientation.z,
    self.current_robot_pose.orientation.w
])
        x,y,_=np.dot(  rotation_matrix,   [x,y,0.0]  )
        x,y=np.dot(  [[0,-1],[1,0]],   [x,y]  )
        x = x + self.current_robot_pose.position.x
        y = y + self.current_robot_pose.position.y
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        
        pose.pose.position.z = 0.14
        
        #pose.pose.orientation =  self.current_robot_pose.orientation # 假設沒有旋轉

        # print(str(pose.pose))
        # print(str(self.current_robot_pose))
        # 將位姿添加到Path訊息中
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        # 發布Path訊息
        self.path_pub.publish(self.path_msg)

        
        

    def scan_callback(self, msg):
        leg_points = self.detect_leg_points(msg)
        if leg_points is not None and len(leg_points) > 0:
            markers = self.publish_leg_markers(msg, leg_points)
            self.publish_vector_arrows(msg.header.frame_id, leg_points)
            self.publish_vector_distance(msg.header.frame_id, leg_points)
            for marker in markers:
                self.marker_publisher.publish(marker)
        else:
            self.clear_markers(msg.header.frame_id)
            self.clear_arrows(msg.header.frame_id)
            self.clear_distance(msg.header.frame_id)
            self.clear_cluster(msg.header.frame_id)
            self.clear_path()
            self.clear_robot_trajectory()
        self.publish_front_arrow(msg.header.frame_id)
        self.publish_Mr_Chou(msg.header.frame_id)

        if hasattr(self, 'target_detected') and self.target_detected:
            if self.target_detected:
                cluster_points, cluster_labels = self.cluster_around_target(msg)
                cluster_markers = self.create_cluster_markers(msg.header.frame_id, cluster_points, cluster_labels)
                for marker in cluster_markers:
                    self.cluster_marker_publisher.publish(marker)
            self.check_target_movement()

            # if distance_to_target < self.shutdown_distance_threshold:
            #     self.get_logger().warn('Distance to target is less than 20cm.')
            #     # self.publish_start_end()  # Publish START and END
            #     self.clear_all_markers(msg.header.frame_id)
            #     temp = Bool()
            #     temp.data = True
            #     # self.start_position_publisher.publish(self.current_robot_pose)
            #     self.go_back_publisher.publish(temp)
            #     # rclpy.shutdown()  
            #     return
    def is_stationary(self, current_position, last_position):
        dx = current_position[0] - last_position[0]
        dy = current_position[1] - last_position[1]
        distance = math.hypot(dx, dy)
        return distance < self.position_tolerance

    def check_target_movement(self):
        now = time.time()
        current_position = self.target_position[:2]  # 只考慮 x, y

        if self.last_target_position is None:
            self.last_target_position = current_position
            self.last_stable_time = now
            return

        if self.is_stationary(current_position, self.last_target_position):
            if now - self.last_stable_time > self.stability_duration:
                self.get_logger().warn('Target position has not moved significantly for 10 seconds. Shutting down.')
                self.shutdown_procedure()
        else:
            # 位置有變動，更新記錄
            self.last_stable_time = now
            self.last_target_position = current_position

    def shutdown_procedure(self):
        frame_id = getattr(self, "latest_frame_id", "base_link")  # fallback 預設為 base_link
        self.clear_all_markers(frame_id)
        temp = Bool()
        temp.data = True
        self.go_back_publisher.publish(temp)
        rclpy.shutdown()
    def detect_leg_points(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
       
        if not hasattr(self, 'target_detected'):
            self.target_detected = False
        
        if not self.target_detected:
        
            valid_mask = (ranges > 0.1) & (ranges < 0.35)
        else:
            
            valid_mask = (ranges < 3.0)
            

        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        self.points = np.column_stack((xs, ys))

        clustering_successful = False
        if len(self.points) > 0:
            clustering = DBSCAN(eps=0.1, min_samples=50).fit(self.points)
            clustering_successful = True

        leg_points = []
        
        
        if self.target_detected and clustering_successful:
            # target_index = self.find_nearest_point_index(points, self.target_position)
            valid_mask = np.linalg.norm(self.points - self.target_position,axis=1) < 0.15
            if np.sum(valid_mask)!=0:
                self.points = self.points[valid_mask]
                # clustering = DBSCAN(eps=0.1, min_samples=3).fit(self.points)  
                cluster_mean = np.mean(self.points, axis=0)
                self.target_odom=np.subtract(cluster_mean,self.target_position)
                self.target_position = cluster_mean
            
            leg_points.append(self.target_position)

        if clustering_successful and (not self.target_detected):   
        # else:
            for label in set(clustering.labels_):
                if label != -1:
                    cluster = self.points[clustering.labels_ == label]
                    pca = PCA(n_components=1)
                    pca.fit(cluster)
                    variance_ratio = pca.explained_variance_ratio_[0]
                    if 0.89 < variance_ratio < 0.91:
                        cluster_mean = np.mean(cluster, axis=0)
                        leg_points.append(cluster_mean)
                        if not self.target_detected:
                            self.target_detected = True
                            self.target_position = cluster_mean
            
        return leg_points if leg_points else []

    def cluster_around_target(self, scan_msg):
        # ranges = np.array(scan_msg.ranges)
        # angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        # valid_mask = (ranges < 3.0)  # 僅考慮3米內的點
        # ranges = ranges[valid_mask]
        # angles = angles[valid_mask]
        # xs = ranges * np.cos(angles)
        # ys = ranges * np.sin(angles)
        # points = np.column_stack((xs, ys))
        target_mask = np.linalg.norm(self.points - self.target_position, axis=1) < 0.15
        cluster_points = self.points[target_mask]
        if len(cluster_points) > 0:
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(cluster_points)
            return cluster_points, clustering.labels_
        else:
            return [], []

    def create_cluster_markers(self, frame_id, points, labels):
        markers = []
        for label in set(labels):
            if label != -1:
                cluster = points[labels == label]
                marker_msg = Marker()
                marker_msg.header.frame_id = frame_id
                marker_msg.type = Marker.SPHERE_LIST
                marker_msg.action = Marker.ADD
                marker_msg.scale.x = 0.05
                marker_msg.scale.y = 0.05
                marker_msg.scale.z = 0.05
                marker_msg.color.a = 1.0
                marker_msg.color.r = 1.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 0.0
                marker_msg.pose.orientation.w = 1.0
                marker_msg.points = [self.point_to_point(p) for p in cluster]
                markers.append(marker_msg)
        return markers   
    
    def publish_vector_arrows(self, frame_id, leg_points):
        for i, center_of_cluster in enumerate(leg_points):
            arrow_marker = Marker()
            arrow_marker.header.frame_id = frame_id
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points.append(Point(x=0.0, y=0.0, z=0.0))
            arrow_marker.points.append(Point(x=center_of_cluster[0], y=center_of_cluster[1], z=0.0))
            arrow_marker.scale.x = 0.02
            arrow_marker.scale.y = 0.04
            arrow_marker.scale.z = 0.0
            arrow_marker.color.a = 1.0
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0
            self.arrow_publisher.publish(arrow_marker)

            # 發佈

    def publish_vector_distance(self, frame_id, leg_points):
        for i, center_of_cluster in enumerate(leg_points):
            arrow_text_marker = Marker()
            arrow_text_marker.header.frame_id = frame_id
            arrow_text_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_text_marker.type =  Marker.TEXT_VIEW_FACING
            arrow_text_marker.action = Marker.ADD
            arrow_text_marker.pose.position.x = center_of_cluster[0] + 0.2
            arrow_text_marker.pose.position.y = center_of_cluster[1] + 0.2
            arrow_text_marker.pose.position.z = 0.0
            arrow_text_marker.pose.orientation.w = 1.0
            arrow_text_marker.scale.x = 1.62
            arrow_text_marker.scale.y = 1.25
            arrow_text_marker.scale.z = 0.06
            arrow_text_marker.color.a = 1.0
            arrow_text_marker.color.r = 1.0
            arrow_text_marker.color.g = 1.0
            arrow_text_marker.color.b = 1.0
            distance = self.calculate_distance([0, 0], center_of_cluster)
            arrow_text_marker.text = '{:.2f}m'.format(distance)
            self.distance_publisher.publish(arrow_text_marker)

            # Update distance data for plotting
            # self.distances.append(distance)
            # self.timestamps.append(datetime.now())
            # self.update_plot(None)
    
    def calculate_distance(self, point1, point2):
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return sqrt(dx ** 2 + dy ** 2)

    def publish_leg_markers(self, scan_msg, leg_points):
        markers = []
        for i, cluster in enumerate(leg_points):
            marker_msg = Marker()
            marker_msg.header.frame_id = scan_msg.header.frame_id
            marker_msg.type = Marker.SPHERE_LIST
            marker_msg.action = Marker.ADD
            marker_msg.scale.x = 0.05
            marker_msg.scale.y = 0.05
            marker_msg.scale.z = 0.05
            marker_msg.color.a = 1.0
            marker_msg.color.r = 1.0
            marker_msg.color.g = 0.0
            marker_msg.color.b = 0.0
            marker_msg.pose.orientation.w = 1.0
            marker_msg.points = [self.point_to_point(cluster)]
            markers.append(marker_msg)
        return markers
    
    # def update_plot(self, frame):
    #     self.ax.clear()
    #     self.ax.plot(self.timestamps, self.distances, label='Distance to Leg')
    #     self.ax.set_xlabel('Time')
    #     self.ax.set_ylabel('Distance (m)')
    #     self.ax.set_title('Real-time Distance to Leg')
    #     self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    #     plt.xticks(rotation=45)
    #     plt.legend()
    #     plt.draw()
    #     plt.pause(0.001)  # Pause briefly to update the plot
    
    def spin(self):
        rclpy.spin(self)
        plt.show()
    def publish_Mr_Chou(self, frame_id):
        model_marker = Marker()
        model_marker.header.frame_id = frame_id
        model_marker.header.stamp = self.get_clock().now().to_msg()
        model_marker.type = Marker.MESH_RESOURCE
        model_marker.action = Marker.ADD

        # 指定模型文件的路徑（此範例使用 .dae 檔案）
        model_marker.mesh_resource = "file:///home/airlab/Downloads/meshes/robot.dae"

        # 設置模型的比例（可根據需要調整）
        model_marker.scale.x = 0.00057
        model_marker.scale.y = 0.00057
        model_marker.scale.z = 0.00057

        # model_marker.pose.position.x=-0.4
        model_marker.pose.position = Point(x=0.2422, y=-0.254, z=-0.07)

        # 計算四元數以沿著 X 軸旋轉 90 度
        roll = math.radians(90)  # 90 度的旋轉
        pitch = 0
        yaw = 0
        q = quaternion_from_euler(roll, pitch, yaw)
        model_marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # 設置模型的顏色（如果模型本身沒有顏色或需要覆蓋）
        model_marker.color.r = 0.66
        model_marker.color.g = 0.66
        model_marker.color.b = 0.66
        model_marker.color.a = 0.8  # 完全不透明

    # 發布模型
        # 發布模型
        self.Mr_Chou_publisher.publish(model_marker)

        
        
    def publish_front_arrow(self, frame_id):
        
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.points.append(Point(x=0.0, y=0.0, z=0.0))
        arrow_marker.points.append(Point(x=0.0, y=-0.3, z=0.0))
        arrow_marker.scale.x = 0.03
        arrow_marker.scale.y = 0.05
        arrow_marker.scale.z = 0.05
        arrow_marker.color.a = 1.0
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 1.0
        self.front_arrow_publisher.publish(arrow_marker)

    def point_to_point(self, point):
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = 0.0
        return p
    
    def clear_markers(self,frame_id):
        delete_msg = Marker()
        delete_msg.header.frame_id = frame_id
        delete_msg.action = Marker.DELETEALL
        self.marker_publisher.publish(delete_msg)
    
    def clear_cluster(self,frame_id):
        delete_msg = Marker()
        delete_msg.header.frame_id = frame_id
        delete_msg.action = Marker.DELETE
        self.cluster_marker_publisher.publish(delete_msg)
        
    def clear_arrows(self,frame_id):
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.DELETE
        self.arrow_publisher.publish(arrow_marker)

    def clear_distance(self,frame_id):
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.DELETE
        self.distance_publisher.publish(arrow_marker)

    def clear_path(self):
    # 創建一個空的 Path 訊息，與之前發布的路徑保持一致
        empty_path_msg = Path()
        empty_path_msg.header.frame_id = 'odom'
        empty_path_msg.header.stamp = self.get_clock().now().to_msg()

        # 清空之前存儲的路徑數據
        self.path_msg.poses.clear()

        # 發布空的 Path 訊息，來清除顯示中的路徑
        self.path_pub.publish(empty_path_msg)
        # self.get_logger().info("Path cleared successfully")

    def clear_robot_trajectory(self):
    # 創建一個空的 Path 訊息來清除 RViz 中的顯示
        empty_trajectory = Path()
        empty_trajectory.header.frame_id = 'odom'
        empty_trajectory.header.stamp = self.get_clock().now().to_msg()

        # 清空儲存的機器人路徑數據
        self.robot_trajectory.poses.clear()

        # 發布這個空的路徑訊息來清除顯示
        self.robot_trajectory_publisher.publish(empty_trajectory)
        
        # self.get_logger().info("Robot trajectory cleared.")

        # 重置 self.robot_trajectory 以開始新的路徑記錄
        self.robot_trajectory = Path()
        self.robot_trajectory.header.frame_id = 'odom'
    

    def clear_all_markers(self, frame_id):
        self.clear_markers(frame_id)
        self.clear_arrows(frame_id)
        self.clear_distance(frame_id)
        self.clear_cluster(frame_id)
        self.clear_path()
        self.clear_robot_trajectory()  


    # def destroy_node(self):
    #     self.clear_all_markers('base_link')  # 清除所有標記
    #     super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    leg_detection_node = LegDetectionNode()
    rclpy.spin(leg_detection_node)
    leg_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
