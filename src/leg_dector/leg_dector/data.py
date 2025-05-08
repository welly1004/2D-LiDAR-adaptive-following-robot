import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from geometry_msgs.msg import Twist, Point, PointStamped
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import time
import math


class DistancePlotter(Node):
    def __init__(self):
        super().__init__('distance_plotter')

        # 訂閱主題
        self.center_subscription = self.create_subscription(
            Marker,
            '/leg_center',
            self.leg_center_callback,
            10
        )
        self.angle_subscriber = self.create_subscription(
            Marker,
            '/robot_to_leg_vector',
            self.vector_arrows_callback,
            10
        )
        self.position_subscriber = self.create_subscription(
            Marker,
            '/robot_to_following_position_vector',
            self.following_pos_callback,
            10
        )

        # 初始化變數
        self.start_time = None
        self.has_started = False
        self.times = []
        self.distances_to_leg_center = []
        self.angle_distances = []
        self.following_pos = []
        self.angle_degrees = 0
        self.distance_to_following_pos = 0.0

        # 新增資料是否準備好旗標
        self.angle_ready = False
        self.following_pos_ready = False

        self.global_frame = 'odom'
        self.robot_frame = 'rplidar_link'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 創建圖表
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 9))  
        self.figure.tight_layout(pad=3.0)  
        plt.ion()  

    def following_pos_callback(self, msg):
        if len(msg.points) < 2:
            self.get_logger().warn("未收到足夠的 /robot_to_following_position_vector 點")
            return

        x, y = msg.points[1].x, msg.points[1].y
        local_coords = self.transform_to_local_frame(x, y)
        
        if local_coords:
            local_x, local_y = local_coords
            self.distance_to_following_pos = math.sqrt(local_x**2 + local_y**2)
            self.following_pos_ready = True  # 資料就緒

            self.get_logger().info(
                f"轉換後的追隨目標位置: x={local_x:.2f}, y={local_y:.2f}, 距離={self.distance_to_following_pos:.2f}"
            )
        else:
            self.get_logger().warn("追隨目標座標轉換失敗，local_coords 為 None")
            self.distance_to_following_pos = None
            self.following_pos_ready = False

    def vector_arrows_callback(self, msg):
        if len(msg.points) < 2:
            self.get_logger().warn("not receive /robot_to_leg_vector message")
            return

        x, y = msg.points[1].x, msg.points[1].y
        magnitude_a = math.sqrt(x ** 2 + y ** 2)
        magnitude_b = math.sqrt(0 ** 2 + 1 ** 2)
        if magnitude_a != 0:
            dot_product = x * 0 + y * -1
            angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))
            self.angle_degrees = round(math.degrees(angle_radians), 2)
            if x < 0:
                self.angle_degrees *= -1

            self.angle_ready = True  # 資料就緒
            self.get_logger().info(f"Angle between the target point and the robot : {self.angle_degrees}")

    def transform_to_local_frame(self, x, y):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.global_frame
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point.x = x
            point_stamped.point.y = y
            point_stamped.point.z = 0.0

            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.global_frame,
                rclpy.time.Time()
            )

            local_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return (local_point.point.x, local_point.point.y)

        except TransformException as ex:
            self.get_logger().warning(f"無法轉換座標: {ex}")
            return None

    def leg_center_callback(self, msg):
        if len(msg.points) == 0:
            self.get_logger().warn("not receive /leg_center message")
            return

        # 若尚未準備好角度與追隨位置資料，不記錄也不繪圖
        if not self.angle_ready or not self.following_pos_ready:
            self.get_logger().info("等待角度與追隨位置資料初始化中...")
            return

        # 若還沒開始記錄，現在開始
        if not self.has_started:
            self.start_time = time.time()
            self.has_started = True
            self.get_logger().info("Started recording data...")

        x, y = msg.points[0].x, msg.points[0].y
        distance_to_leg_center = math.sqrt(x**2 + y**2)

        elapsed_time = time.time() - self.start_time
        self.times.append(elapsed_time)
        self.distances_to_leg_center.append(distance_to_leg_center)
        self.angle_distances.append(self.angle_degrees)
        self.following_pos.append(self.distance_to_following_pos)

        # 更新圖表
        self.ax1.clear()
        self.ax1.plot(self.times, self.distances_to_leg_center, label='Distance to leg', color='blue')
        self.ax1.set_xlabel('Time(s)')
        self.ax1.set_ylabel('Distance(m)')
        self.ax1.set_title('Robot to leg distance.')
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.plot(self.times, self.angle_distances, label='Degree to leg', color='orange')
        self.ax2.set_xlabel('Time(s)')
        self.ax2.set_ylabel('Angle(Degree)')
        self.ax2.set_title('Robot to leg angle.')
        self.ax2.legend()

        self.ax3.clear()
        self.ax3.plot(self.times, self.following_pos, label='Distance to following position', color='green')
        self.ax3.set_xlabel('Time(s)')
        self.ax3.set_ylabel('Distance(m)')
        self.ax3.set_title('Robot to following position distance.')
        self.ax3.legend()

        plt.draw()
        plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)
    node = DistancePlotter()
    rclpy.spin(node)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
