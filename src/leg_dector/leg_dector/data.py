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
        self.start_time = None  # 延遲設定啟動時間
        self.has_started = False  # 是否開始記錄
        self.times = []  # 紀錄時間的列表
        self.distances_to_leg_center = []  # 到 /leg_center 的距離
        self.angle_distances = []  # 角度變化數據
        self.following_pos = []  # 到 /robot_to_following_position_vector 的距離
        self.angle_degrees = 0  # 當前計算的角度（以度為單位）
        self.distance_to_following_pos = 0.0  # 當前計算的距離

        # 創建圖表
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 9))  
        self.figure.tight_layout(pad=3.0)  
        plt.ion()  

    def following_pos_callback(self, msg):
        if len(msg.points) < 2:
            self.get_logger().warn("not receive /robot_to_following_position_vector message")
            return

        x, y = msg.points[1].x, msg.points[1].y
        self.distance_to_following_pos = math.sqrt(x**2 + y**2)  

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

            self.get_logger().info(f"Angle between the target point and the robot : {self.angle_degrees}")
            if x < 0:
                self.angle_degrees *= -1

    def leg_center_callback(self, msg):
        if len(msg.points) == 0:
            self.get_logger().warn("not receive /leg_center message")
            return

        # **如果還沒開始記錄，現在開始**
        if not self.has_started:
            self.start_time = time.time()  
            self.has_started = True  
            self.get_logger().info("Started recording data...")

        # 計算到 /leg_center` 的距離
        x, y = msg.points[0].x, msg.points[0].y
        distance_to_leg_center = math.sqrt(x**2 + y**2)  

        # 紀錄時間和距離數據
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
