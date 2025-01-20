import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import pi, atan2
import threading

# 텍스트 파일에서 지도 읽기
def read_map(file_path):
    with open(file_path, 'r') as f:
        return [list(map(int, line.strip().split())) for line in f]
    
# 맵과 경로를 시각화하는 함수
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

    ax.set_title('A* Path Visualization')
    ani = animation.FuncAnimation(fig, update, interval=100, blit=True)
    ax.legend()
    plt.show()

# Astar 노드 정의
class AStarROS2Node(Node):
    def __init__(self, grid, start, goal, shared_data):
        super().__init__('astar_ros2')
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.subscription = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        
        self.pose = None
        self.grid = grid
        self.start = start
        self.goal = goal
        self.shared_data = shared_data

        self.path = self.astar(grid, start, goal)
        self.current_index = 0

        if self.path is None:
            self.get_logger().warn('경로를 찾을 수 없습니다.')
            return
        
        self.shared_data['path'] = self.path
        self.shared_data['explored_nodes'] = []
        self.shared_data['turtle_pose'] = []

        # 주기적인 타이머를 설정하여 이동을 처리
        self.timer = self.create_timer(0.1, self.move_turtle)

    # 맨해튼 거리 계산
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # A* 알고리즘
    def astar(self, grid, start, goal):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 상, 하, 좌, 우
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_list:
            _, current_g, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            closed_list.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and 
                    grid[neighbor[0]][neighbor[1]] != 1 and neighbor not in closed_list):

                    tentative_g_score = current_g + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score, tentative_g_score, neighbor))
                        came_from[neighbor] = current
            self.shared_data['explored_nodes'].append(current)
        return None

    def pose_callback(self, msg):
        self.pose = msg
        turtle_pose = (self.pose.y, self.pose.x)  # (y, x)로 위치 변환
        self.shared_data['turtle_pose'] = turtle_pose  # 거북이 위치를 shared_data에 추가
        self.get_logger().info(f'현재 위치: ({self.pose.x}, {self.pose.y}, {self.pose.theta})')

    def move_turtle(self):
        if self.pose is None or self.path is None or self.current_index >= len(self.path):
            return

        move_cmd = Twist()
        
        # 목표 지점까지 가는 경로에서 현재 목표 점을 추출
        target = self.path[self.current_index]

        # 목표 위치와 현재 위치 간의 거리와 각도를 계산
        dx = target[1] - self.pose.x
        dy = target[0] - self.pose.y
        distance = (dx ** 2 + dy ** 2) ** 0.5

        # 목표 지점에 가까워지면 다음 경로 점으로 이동
        if distance < 0.1:
            self.current_index += 1  # 경로상 다음 점으로 이동

            # 모든 경로를 완료했을 때
            if self.current_index >= len(self.path):
                self.get_logger().info('목표 지점에 도착했습니다. 실행을 멈춥니다.')
                self.get_logger().info(f'경로 길이는 {len(self.path)}입니다.')
                rclpy.shutdown()
                return
            
        # 각도 계산 (단순히 2D 회전)
        angle_to_goal = atan2(dy, dx)
        
        # 각도 차이를 -pi에서 pi 사이로 정규화
        angle_diff = angle_to_goal - self.pose.theta
        if angle_diff > pi:
            angle_diff -= 2 * pi
        elif angle_diff < -pi:
            angle_diff += 2 * pi
        
        move_cmd.angular.z = angle_diff * 4.0  # 현재 회전 각도에서 목표 각도 계산
        move_cmd.linear.x = 1.0 if abs(angle_diff) < 0.1 else 0.0
        
        self.publisher.publish(move_cmd)


def main(args=None):
    rclpy.init(args=args)

    grid = read_map('/home/soeun/map/src/map/map/map.txt') 
    start = (0, 0)  # 시작점
    goal = (5, 5)  # 목표점

    shared_data = {
        'explored_nodes': [],  
        'path': [],
        'turtle_pose': None, 
    }

    node = AStarROS2Node(grid, start, goal, shared_data)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

        # 맵과 경로 시각화
    visualize_map_and_path(grid, shared_data)

    # rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()
