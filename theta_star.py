"""
Theta* path planning

Adapted from A* implementation by:
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

Theta* implementation adds any-angle path planning capabilities
"""

import math
import matplotlib.pyplot as plt

show_animation = True


class ThetaStarPlanner:
    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y, sc_x, sc_y, Cf, Ct, n, M):
        """
        Initialize grid map for Theta* planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

        self.fc_x = fc_x
        self.fc_y = fc_y
        self.tc_x = tc_x
        self.tc_y = tc_y
        self.sc_x = sc_x
        self.sc_y = sc_y

        self.Delta_C1 = 0.2
        self.Delta_C2 = 1
        self.Delta_C3 = 0.05

        self.costPerGrid = 1
        
        # Store user input parameters
        self.Cf = Cf
        self.time_cost_level = Ct
        self.n = n
        self.M = M
        
        # Aircraft specific parameters
        self.aircraft_params = {
            'A321': {
                'delta_F': 54,
                'Ct': {'L': 10, 'M': 15, 'H': 20},
                'C': 1800,
                'p': 200
            },
            'A339': {
                'delta_F': 84,
                'Ct': {'L': 15, 'M': 21, 'H': 27},
                'C': 2000,
                'p': 300
            },
            'A359': {
                'delta_F': 90,
                'Ct': {'L': 20, 'M': 27, 'H': 34},
                'C': 2500,
                'p': 350
            }
        }

    class Node:
        def __init__(self, x, y, cost, parent=None):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent = parent
            
        def __str__(self):
            return f"({self.x}, {self.y})"

    def line_of_sight(self, node1, node2):
        """Check if there's a direct line of sight between two nodes"""
        x1 = self.calc_grid_position(node1.x, self.min_x)
        y1 = self.calc_grid_position(node1.y, self.min_y)
        x2 = self.calc_grid_position(node2.x, self.min_x)
        y2 = self.calc_grid_position(node2.y, self.min_y)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x = x1
        y = y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            # Convert real coordinates back to grid indices
            grid_x = self.calc_xy_index(x, self.min_x)
            grid_y = self.calc_xy_index(y, self.min_y)
            
            # Check if point is within bounds and not in obstacle
            if not (0 <= grid_x < self.x_width and 0 <= grid_y < self.y_width) or \
               self.obstacle_map[grid_x][grid_y]:
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True

    def calc_path_cost(self, node1, node2):
        """Calculate path cost between two nodes considering cost areas"""
        x1 = self.calc_grid_position(node1.x, self.min_x)
        y1 = self.calc_grid_position(node1.y, self.min_y)
        x2 = self.calc_grid_position(node2.x, self.min_x)
        y2 = self.calc_grid_position(node2.y, self.min_y)
        
        base_cost = math.hypot(x2 - x1, y2 - y1) * self.costPerGrid
        
        # Add cost modifiers based on areas
        # Note: This is a simplified version. For more accuracy, you might want to
        # check if the line passes through cost areas
        if (x2 in self.tc_x and y2 in self.tc_y):
            base_cost += self.Delta_C1 * base_cost
        if (x2 in self.fc_x and y2 in self.fc_y):
            base_cost += self.Delta_C2 * base_cost
        if (x2 in self.sc_x and y2 in self.sc_y):
            base_cost -= self.Delta_C3 * base_cost
            
        return base_cost

    def planning(self, sx, sy, gx, gy):
        """
        Theta* path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                             self.calc_xy_index(sy, self.min_y), 0.0)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                            self.calc_xy_index(gy, self.min_y), 0.0)

        open_set = {self.calc_grid_index(start_node): start_node}
        closed_set = {}

        while open_set:
            current = min(open_set.values(),
                        key=lambda n: n.cost + self.calc_heuristic(n, goal_node))
            c_id = self.calc_grid_index(current)

            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                        self.calc_grid_position(current.y, self.min_y), "xc")
                plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent = current.parent
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            # Expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                               current.y + self.motion[i][1],
                               current.cost + self.motion[i][2],
                               current)

                n_id = self.calc_grid_index(node)

                if not self.verify_node(node) or n_id in closed_set:
                    continue

                # This is where Theta* differs from A*
                # Instead of just checking the current node as parent,
                # we see if we can make a straight line to the grandparent
                if current.parent and self.line_of_sight(current.parent, node):
                    # If we have line of sight to grandparent, calculate new cost
                    new_cost = current.parent.cost + self.calc_path_cost(current.parent, node)
                    if n_id not in open_set or new_cost < node.cost:
                        node.cost = new_cost
                        node.parent = current.parent
                        open_set[n_id] = node
                else:
                    # No line of sight, fallback to standard A* behavior
                    if n_id not in open_set:
                        open_set[n_id] = node
                    elif node.cost < open_set[n_id].cost:
                        open_set[n_id] = node

        rx, ry = self.extract_path(goal_node)
        return rx, ry

    def extract_path(self, goal_node):
        rx, ry = [], []
        current = goal_node
        while current:
            rx.append(self.calc_grid_position(current.x, self.min_x))
            ry.append(self.calc_grid_position(current.y, self.min_y))
            current = current.parent
        return rx[::-1], ry[::-1]

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d * self.costPerGrid

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y:
            return False
        if px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)]
                           for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1],
                 [0, 1, 1],
                 [-1, 0, 1],
                 [0, -1, 1],
                 [-1, -1, math.sqrt(2)],
                 [-1, 1, math.sqrt(2)],
                 [1, -1, math.sqrt(2)],
                 [1, 1, math.sqrt(2)]]
        return motion


def main():
    print("Theta* path planning demo")

    # Get user inputs
    Cf = float(input("Enter the cost of fuel (Cf): "))
    
    while True:
        time_cost = input("Enter time cost (L/M/H): ").upper()
        if time_cost in ['L', 'M', 'H']:
            Ct = {'L': 0.8, 'M': 1.0, 'H': 1.2}[time_cost]
            break
        print("Invalid input. Please enter L, M, or H.")
    
    while True:
        try:
            n = int(input("Enter number of passengers: "))
            if n > 0:
                break
            print("Number of passengers must be positive.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            M = int(input("Enter maximum number of flights: "))
            if M > 0:
                break
            print("Maximum number of flights must be positive.")
        except ValueError:
            print("Please enter a valid integer.")

    # start and goal position
    sx = 0.0
    sy = 10.0
    gx = 60.0
    gy = 25.0
    grid_size = 1
    robot_radius = 1.0

    # Set obstacle positions
    ox, oy = [], []
    for i in range(-10, 70):  # bottom border
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):  # right border
        ox.append(70.0)
        oy.append(i)
    for i in range(-10, 70):  # top border
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 60):  # left border
        ox.append(-10.0)
        oy.append(i)

    for i in range(0, 20):  # internal obstacles
        ox.append(20.0)
        oy.append(i)
    
    for i in range(30, 55):
        ox.append(10.0)
        oy.append(i)
    
    for i in range(0, 20):
        ox.append(30.0)
        oy.append(i)

    # Set cost intensive areas
    tc_x, tc_y = [], []
    for i in range(10, 20):
        for j in range(10, 45):
            tc_x.append(i)
            tc_y.append(j)
    
    fc_x, fc_y = [], []
    for i in range(30, 45):
        for j in range(10, 35):
            fc_x.append(i)
            fc_y.append(j)

    # Set jet stream
    sc_x, sc_y = [], []
    for i in range(0, 60):
        for j in range(-2, 3):
            sc_x.append(i)
            sc_y.append(j)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(fc_x, fc_y, "oy")
        plt.plot(tc_x, tc_y, "or")
        plt.plot(sc_x, sc_y, "og")
        plt.grid(True)
        plt.axis("equal")

    theta_star = ThetaStarPlanner(ox, oy, grid_size, robot_radius,
                                fc_x, fc_y, tc_x, tc_y, sc_x, sc_y,
                                Cf, Ct, n, M)
    rx, ry = theta_star.planning(sx, sy, gx, gy)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()