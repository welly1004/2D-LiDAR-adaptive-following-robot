# LiDAR Human Following Robot in ROS2

本專案基於 **TurtleBot4 Lite**，在 **ROS2 Galactic** 使用 **RPLIDAR A1 光達** 實現自適應跟隨機器人。  
系統包含 人腿偵測、最佳跟隨位置評估、局部規劃 (DWA)、復位全域規劃 (QRRT star)、跟隨數據統計 五大功能，並透過 **RViz2** 進行可視化。
<img width="1514" height="825" alt="image" src="https://github.com/user-attachments/assets/e74ebd31-b39c-417a-9b0f-b20efa379376" />

---

## 硬體配置

- TurtleBot4 Lite  
- RPLIDAR A1  
- Ubuntu 20.04 + ROS2 Galactic  

---

## Package 名稱

`leg_dector`

---

## 節點說明

### 1. 腿部檢測 (leg_detector.py)

- **功能**：從光達資料中檢測人腿，輸出腿部位置。  
- **輸入**：  
  - `/scan`  
- **輸出 Topic**：  
  - `/robot_front_vector` → 機器人的朝向向量  
  - `/leg_marker` → 標記系統檢測到的腿部位置  
  - `/robot_to_leg_vector` → 機器人到腿部的向量  
  - `/leg_distance` → 機器人到腿部的距離  
  - `/leg_center` → 腿部中心點位置  
  - `/go_back` → 結束跟隨模式布林值 (尚未更新)  

---

### 2. 最佳跟隨位置評估 (following_position.py)

- **功能**：以腿部為中心生成候選跟隨位置，並透過目標函數算出最佳跟隨位置。  
- **輸入**：  
  - `/scan`  
  - `/leg_center`  
- **輸出 Topic**：  
  - `/following_pos_marker` → 跟隨候選點 (顏色越淺分數越高)  
  - `/score_text_markers` → 每個跟隨候選點分數  
  - `/robot_to_following_position_vector` → 機器人到最佳跟隨位置向量  
  - `/long_axis_vector` → 跟隨目標移動向量  
  - `/best_point_vector` → 跟隨目標到最佳跟隨位置向量  
  - `/next_leg_marker` → 預測下個時刻腿部位置  

---

### 3. 局部路徑規劃 (dwa_path_planner.py)

- **功能**：基於 **DWA (Dynamic Window Approach)** 實現局部避障與跟隨目標。  
- **輸入**：  
  - `/scan`  
  - `/robot_to_following_position_vector`  
- **輸出 Topic**：  
  - `/dwa_paths` → 紫色為機器人選出的最佳路徑  
  - `/cmd_vel` → 控制馬達移動  

---

### 4. 復位模式 (go_back.py) (尚未更新)

- **功能**：接受到結束跟隨的布林值，開始使用 **QRRT*** 回到跟隨起點。  
- **輸入**：  
  - `/map`  
  - `/go_back`  
- **輸出 Topic**：  
  - `/cmd_vel` → 控制馬達移動  

---

### 5. 跟隨過程中數據統計 (data.py)

- **功能**：統計跟隨過程中各項數據。  

---

## 啟動流程

### 1. 建立 ROS2 工作空間

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. 複製本專案

```bash
git clone https://github.com/welly1004/2D-LiDAR-adaptive-following-robot.git
```

### 3. 編譯

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 4. 執行

#### 腿部檢測、跟隨點評估

```bash
ros2 run leg_dector leg_dection
ros2 run leg_dector following_position
```

#### 確認最佳跟隨點後再開啟局部路徑規劃

```bash
ros2 run leg_dector dwa_planner
```

#### 數據統計

```bash
ros2 run leg_dector data
```

#### 如需復位模式 (細節尚未更新)

```bash
ros2 run leg_dector go_back
```
