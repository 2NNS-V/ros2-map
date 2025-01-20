import threading
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import pi, atan2

# 텍스트 파일에서 지도 읽기
def read_map(file_path):
    with open(file_path, 'r') as f:
        return [list(map(int, line.strip().split())) for line in f]

# 노드 정의
class DFSNode(Node):
    def __init__(self, grid, start, goal, shared_data):
        super().__init__('dfs_ros2')
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.subscription = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

        self.pose = None
        self.grid = grid
        self.start = start
        self.goal = goal
        self.shared_data = shared_data

        # 경로 탐색
        self.path = self.dfs(grid, start, goal)
        self.current_index = 0

        if not self.path:
            self.get_logger().warn('경로를 찾을 수 없습니다.')
            return

        # 탐색 상태 공유
        self.shared_data['path'] = self.path
        self.shared_data['explored_nodes'] = []
        self.shared_data['turtle_pose'] = []

        # 타이머 설정
        self.timer = self.create_timer(0.1, self.move_turtle)

    def dfs(self, grid, start, goal):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 상, 하, 좌, 우
        stack = [(start, [start])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # 목표에 도달하면 경로 반환
            if current == goal:
                return path

            # 상하좌우로 이동
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # 맵 범위 내에 있는지 확인하고 벽을 피함
                if (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])
                        and grid[neighbor[0]][neighbor[1]] != 1  # 벽이 아니고
                        and neighbor not in visited):  # 방문하지 않은 곳
                    stack.append((neighbor, path + [neighbor]))

            # 실시간 탐색 상태 공유
            self.shared_data['explored_nodes'].append(current)
            print(current)

        return []


    def pose_callback(self, msg):
        self.pose = msg
        turtle_pose = (self.pose.y, self.pose.x)  # (y, x)로 위치 변환
        self.shared_data['turtle_pose'] = turtle_pose  # 거북이 위치를 shared_data에 추가
        self.get_logger().info(f'현재 위치: ({self.pose.x}, {self.pose.y}, {self.pose.theta})')

    def move_turtle(self):
        if self.pose is None or self.current_index >= len(self.path):
            return

        target = self.path[self.current_index]
        dx = target[1] - self.pose.x
        dy = target[0] - self.pose.y

        distance = (dx ** 2 + dy ** 2) ** 0.5

        move_cmd = Twist()

        if distance < 0.1:
            self.current_index += 1
            if self.current_index >= len(self.path):
                self.get_logger().info('목표 지점에 도착했습니다. ')
                self.get_logger().info(f'경로 길이는 {len(self.path)}입니다.')
                rclpy.shutdown()
                return

        angle_to_goal = atan2(dy, dx)
        angle_diff = angle_to_goal - self.pose.theta
        if angle_diff > pi:
            angle_diff -= 2 * pi
        elif angle_diff < -pi:
            angle_diff += 2 * pi

        move_cmd.angular.z = angle_diff * 5.0
        move_cmd.linear.x = 1.0 if abs(angle_diff) < 0.1 else 0.0

        self.publisher.publish(move_cmd)

# 지도 및 탐색 상태 시각화
def visualize_map_and_path(grid, shared_data):
    grid = np.array(grid)
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', origin='lower')

    path_line, = ax.plot([], [], 'b-o', label='Path', markersize=5)
    explored_scatter, = ax.plot([], [], 'ro', label='Explored Nodes', markersize=3)
    turtle_position, = ax.plot([], [], 'go', markersize=8, label='Turtle Position') 

    def update(_):
        current_path = shared_data.get('path', [])
        explored_nodes = shared_data['explored_nodes']
        turtle_pose = shared_data.get('turtle_pose', None)  # 현재 거북이 위치

        # 업데이트된 탐색된 노드와 경로를 시각화
        if explored_nodes:
            explored_x = [n[1] for n in explored_nodes]
            explored_y = [n[0] for n in explored_nodes]
            explored_scatter.set_data(explored_x, explored_y)

        # 현재 경로 표시
        if current_path:
            path_x = [p[1] for p in current_path]
            path_y = [p[0] for p in current_path]
            path_line.set_data(path_x, path_y)
        
        if turtle_pose:
            turtle_position.set_data(turtle_pose[1], turtle_pose[0])
            
        return explored_scatter, path_line, turtle_position

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True)
    ax.legend()
    plt.show()

def main(args=None):
    rclpy.init(args=args)

    # 지도 파일 경로 및 초기화
    grid = read_map('/home/soeun/map/src/map/map/map.txt')
    start = (0, 0)
    goal = (5, 5)

    shared_data = {
        'explored_nodes': [],  
        'path': [],
        'turtle_pose': None, 
    }

    # ROS 2 노드 실행
    node = DFSNode(grid, start, goal, shared_data)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 지도 시각화 실행
    visualize_map_and_path(grid, shared_data)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
