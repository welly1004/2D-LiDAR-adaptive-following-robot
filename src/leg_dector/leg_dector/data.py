import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import time
import math
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy

class DistancePlotter(Node):
    def __init__(self):
        super().__init__('distance_plotter')

        # 訂閱 /leg_center 和 /robot_to_leg_vector
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

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        )
        self.best_point_vector_sub = self.create_subscription(
            Marker,
            'best_point_vector',
            self.best_point_vector_callback,
            10
        )

        self.long_axis_vector_sub = self.create_subscription(
            Marker,
            'long_axis_vector',
            self.long_axis_vector_callback,
            10
        )

        # 初始化變數
        self.start_time = None
        self.has_started = False
        self.times = []
        self.distances_to_leg_center = []
        self.angle_distances = []

        self.angle_degrees = 0
        self.angle_ready = False

        self.dynamic_target_distance = None
        self.dynamic_target_angle = None

        self.dynamic_target_distances = []
        self.dynamic_target_angles = []
        # 創建圖表
        self.figure, axes = plt.subplots(3, 2, figsize=(24, 18))
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6 = axes.flatten()
        self.figure.subplots_adjust(hspace=0.4, wspace=0.3)  # 控制垂直與水平間距

        
        plt.ion()

        self.last_save_time = time.time()
        self.save_interval = 2.0
        self.save_dir = '/home/htm/Desktop/2D-LiDAR-adaptive-following-robot-main/data'
        os.makedirs(self.save_dir, exist_ok=True)

        # 記錄線速度與角速度
        self.linear_velocities = []
        self.angular_velocities = []

        self.last_cmd_vel_time = time.time()
        self.current_linear = 0.0
        self.current_angular = 0.0

        # 定時回呼
        self.timer = self.create_timer(0.1, self.timer_callback)
    def best_point_vector_callback(self, msg):
        if len(msg.points) >= 2:
            p1 = msg.points[0]
            p2 = msg.points[1]
            self.dynamic_target_distance = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

            # 儲存向量，用於之後計算夾角
            self.best_vector = (p2.x - p1.x, p2.y - p1.y)
        else:
            self.dynamic_target_distance = None
            self.best_vector = None

    def long_axis_vector_callback(self, msg):
        if len(msg.points) >= 2:
            p1 = msg.points[0]
            p2 = msg.points[1]
            self.long_axis_vector = (p2.x - p1.x, p2.y - p1.y)
        else:
            self.long_axis_vector = None

    def vector_arrows_callback(self, msg):
        """ 處理角度資訊 """
        if len(msg.points) < 2:
            self.get_logger().warn("未收到 /robot_to_leg_vector 點")
            return

        x, y = msg.points[1].x, msg.points[1].y
        magnitude = math.sqrt(x**2 + y**2)

        if magnitude != 0:
            dot = y * -1  # 因為 y 軸朝上，正方向朝前
            angle_radians = math.acos(dot / (magnitude * 1.0))
            self.angle_degrees = round(math.degrees(angle_radians), 2)
            if x < 0:
                self.angle_degrees *= -1
            self.angle_ready = True
            # self.get_logger().info(f"Robot 與 leg 的角度：{self.angle_degrees:.2f} 度")

    def cmd_vel_callback(self, msg):
        """ 更新線速度與角速度 """
        self.current_linear = msg.linear.x
        self.current_angular = msg.angular.z
        self.last_cmd_vel_time = time.time()

    def timer_callback(self):
        """ 定時回呼，處理速度更新 """
        if not self.has_started:
            return

        current_time = time.time()
        elapsed = current_time - self.start_time

        # 判斷線速度和角速度
        if current_time - self.last_cmd_vel_time > 0.2:
            linear = 0.0
            angular = 0.0
        else:
            linear = self.current_linear
            angular = self.current_angular

        self.linear_velocities.append((elapsed, linear))
        self.angular_velocities.append((elapsed, angular))

    def leg_center_callback(self, msg):
        """ 處理 leg_center 點，並更新圖表 """
        if len(msg.points) == 0:
            self.get_logger().warn("未收到 /leg_center 點")
            return

        if not self.angle_ready:
            self.get_logger().info("等待角度資料初始化中...")
            return

        if not self.has_started:
            self.start_time = time.time()
            self.has_started = True
            self.get_logger().info("開始記錄資料...")

        x, y = msg.points[0].x, msg.points[0].y
        print(x,y)
        distance = math.sqrt(x**2 + y**2)
        print(distance)
        elapsed = time.time() - self.start_time

        self.times.append(elapsed)
        self.distances_to_leg_center.append(distance)
        self.angle_distances.append(self.angle_degrees)
        # 更新動態目標角度（夾角）
        if hasattr(self, 'best_vector') and self.best_vector and hasattr(self, 'long_axis_vector') and self.long_axis_vector:
            dot = self.best_vector[0] * self.long_axis_vector[0] + self.best_vector[1] * self.long_axis_vector[1]
            cross = self.best_vector[0] * self.long_axis_vector[1] - self.best_vector[1] * self.long_axis_vector[0]
            mag1 = math.sqrt(self.best_vector[0]**2 + self.best_vector[1]**2)
            mag2 = math.sqrt(self.long_axis_vector[0]**2 + self.long_axis_vector[1]**2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
                angle = math.degrees(math.acos(cos_angle))
                if cross < 0:
                    angle *= -1  # 在右邊，設為負角度
                self.dynamic_target_angle = round(angle, 2)

        # 儲存時間對應的動態目標值
        self.dynamic_target_distances.append(self.dynamic_target_distance if self.dynamic_target_distance is not None else 0.0)
        self.dynamic_target_angles.append(self.dynamic_target_angle if self.dynamic_target_angle is not None else 0.0)

        # 圖1：實際距離 + 動態距離曲線（虛線）
        self.ax1.clear()
        self.ax1.plot(self.times, self.distances_to_leg_center, label='Distance to Leg', color='blue')
        if self.dynamic_target_distances:
            self.ax1.plot(self.times, self.dynamic_target_distances, linestyle='--', color='gray', label='Optomal Position')
        self.ax1.set_title("Actual Distance of Robot to Leg")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Distance (m)")
        self.ax1.legend()

        # 圖2：實際角度 + 動態角度曲線（虛線）
        self.ax2.clear()
        self.ax2.plot(self.times, self.angle_distances, label='Angle to Leg', color='orange')
        if self.dynamic_target_angles:
            self.ax2.plot(self.times, self.dynamic_target_angles, linestyle='--', color='gray', label='Optimal Position')
        self.ax2.set_title("Actual Angle of Robot to Leg")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Angle (deg)")
        self.ax2.legend()


        # 圖3：距離誤差 = 實際 - 動態目標距離
        self.ax3.clear()
        if len(self.dynamic_target_distances) == len(self.distances_to_leg_center):
            distance_errors = [
                actual - target
                for actual, target in zip(self.distances_to_leg_center, self.dynamic_target_distances)
            ]
            self.ax3.plot(self.times, distance_errors, label='Distance Error ', color='blue')
        self.ax3.set_title("Error Distance of Robot to Leg")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Error (m)")
        self.ax3.legend()

        # 圖4：角度誤差 = 實際 - 動態目標角度
        self.ax4.clear()
        if len(self.dynamic_target_angles) == len(self.angle_distances):
            angle_errors = [
                abs(actual - target)
                for actual, target in zip(self.angle_distances, self.dynamic_target_angles)
            ]
            self.ax4.plot(self.times, angle_errors, label='Angle Error ', color='orange')
        self.ax4.set_title("Error Angle of Robot to Leg")
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Error (deg)")
        self.ax4.legend()



        # 圖5：線速度
        if self.linear_velocities:
            t_lin, v_lin = zip(*self.linear_velocities)
            self.ax5.clear()
            self.ax5.plot(t_lin, v_lin, label='Linear Velocity', color='purple')
            self.ax5.set_title("Linear Velocity of Robot ")
            self.ax5.set_xlabel("Time (s)")
            self.ax5.set_ylabel("Linear vel (m/s)")
            self.ax5.legend()

        # 圖6：角速度
        if self.angular_velocities:
            t_ang, v_ang = zip(*self.angular_velocities)
            self.ax6.clear()
            self.ax6.plot(t_ang, v_ang, label='Angular Velocity', color='red')
            self.ax6.set_title("Angular Velocity of Robot ")
            self.ax6.set_xlabel("Time (s)")
            self.ax6.set_ylabel("Angular vel (rad/s)")
            self.ax6.legend()

        plt.draw()
        plt.pause(0.01)

        # 定時儲存圖表
        now = time.time()
        if now - self.last_save_time >= self.save_interval:
            save_path = os.path.join(self.save_dir, f'distance_plot_{int(now)}.png')
            self.figure.savefig(save_path)
            self.get_logger().info(f'圖表已儲存到 {save_path}')
            self.last_save_time = now

def main(args=None):
    rclpy.init(args=args)
    node = DistancePlotter()
    rclpy.spin(node)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
