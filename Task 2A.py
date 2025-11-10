"""
A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

Modified for Task A2 - Changing Environment
"""

import math
import matplotlib.pyplot as plt
import random

show_animation = True

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y):
        """
        Initialize grid map for a star planning

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
        
        # Only fuel-consuming area remains
        self.Delta_C2 = 1  # fuel-consuming area modifier
        self.costPerGrid = 1

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        """
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # show graph
            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Total Trip time required -> ", current.cost)
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2] * self.costPerGrid, c_id)
                
                # Add cost in fuel-consuming area
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        node.cost = node.cost + self.Delta_C2 * self.motion[i][2]

                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        d = d * self.costPerGrid
        return d

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

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
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
        # Disable diagonal movement - only up, down, left, right
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]
        return motion


def generate_random_points(min_dist=40, world_size=70):
    """Generate random start and end points with minimum distance"""
    while True:
        sx = random.uniform(0, world_size)
        sy = random.uniform(0, world_size)
        gx = random.uniform(0, world_size)
        gy = random.uniform(0, world_size)
        
        dist = math.hypot(gx - sx, gy - sy)
        if dist >= min_dist:
            return sx, sy, gx, gy


def generate_obstacles(world_size=70, density=0.05, safe_zones=None):
    """Generate random obstacles with reasonable density, avoiding safe zones"""
    ox, oy = [], []
    num_obstacles = int(world_size * world_size * density)
    
    for _ in range(num_obstacles):
        while True:
            x = random.uniform(0, world_size)
            y = random.uniform(0, world_size)
            
            # Check if point is in any safe zone
            in_safe_zone = False
            if safe_zones:
                for (safe_x, safe_y, safe_r) in safe_zones:
                    if math.hypot(x - safe_x, y - safe_y) <= safe_r:
                        in_safe_zone = True
                        break
            
            if not in_safe_zone:
                ox.append(x)
                oy.append(y)
                break
    
    return ox, oy


def generate_fuel_area(world_size=70, size=40, obstacles=None):
    """Generate random fuel-consuming area that doesn't cover obstacles"""
    while True:
        # Random position for fuel area (considering 70x70 world and 40x40 fuel area)
        fc_x_start = random.uniform(0, world_size - size)
        fc_y_start = random.uniform(0, world_size - size)
        
        fc_x, fc_y = [], []
        for i in range(int(size)):
            for j in range(int(size)):
                fc_x.append(fc_x_start + i)
                fc_y.append(fc_y_start + j)
        
        # Check if fuel area covers too many obstacles
        if obstacles:
            ox, oy = obstacles
            obstacle_count = 0
            for x, y in zip(ox, oy):
                if (fc_x_start <= x <= fc_x_start + size and 
                    fc_y_start <= y <= fc_y_start + size):
                    obstacle_count += 1
            
            # If less than 10% of fuel area is covered by obstacles, accept it
            if obstacle_count < (size * size * 0.1):
                return fc_x, fc_y
        else:
            return fc_x, fc_y


def main():
    print(__file__ + " start the A star algorithm demo!!")

    # Set world parameters - changed to 70x70
    world_size = 70
    grid_size = 1.0
    robot_radius = 1.0
    
    # Generate random start and goal with minimum distance
    sx, sy, gx, gy = generate_random_points(min_dist=40, world_size=world_size)
    
    # Define safe zones around start and goal (no obstacles in these areas)
    safe_radius = 8
    safe_zones = [(sx, sy, safe_radius), (gx, gy, safe_radius)]
    
    # Generate obstacles avoiding safe zones
    ox, oy = generate_obstacles(world_size=world_size, density=0.2, safe_zones=safe_zones)
    
    # Generate fuel-consuming area that doesn't cover too many obstacles
    fc_x, fc_y = generate_fuel_area(world_size=world_size, size=40, obstacles=(ox, oy))

    # Add border obstacles
    for i in range(0, int(world_size) + 1):
        ox.append(i)
        oy.append(0.0)
        ox.append(i)
        oy.append(world_size)
        ox.append(0.0)
        oy.append(i)
        ox.append(world_size)
        oy.append(i)

    if show_animation:
        plt.figure(figsize=(10, 10))
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og", markersize=10, label='Start')
        plt.plot(gx, gy, "xb", markersize=10, label='Goal')
        
        # Plot fuel-consuming area
        plt.plot(fc_x, fc_y, "oy", alpha=0.3, label='Fuel-consuming Area')
        
        # Plot safe zones
        for (safe_x, safe_y, safe_r) in safe_zones:
            circle = plt.Circle((safe_x, safe_y), safe_r, color='green', alpha=0.1)
            plt.gca().add_patch(circle)
        
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-5, world_size + 5)
        plt.ylim(-5, world_size + 5)
        plt.legend()
        plt.title("A* Path Planning with Random Environment (70x70)")

    # Create planner and find path
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius, fc_x, fc_y)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:
        plt.plot(rx, ry, "-r", linewidth=2, label='Path')
        plt.pause(0.001)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()