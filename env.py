from shapely.geometry import LineString, Point, MultiLineString
from shapely import affinity
from shapely import ops
import numpy as np
import matplotlib.pyplot as plt


class WormFoodEnv:
    def __init__(self, food_coord, num_parts=20, theta_max=25):
        self.num_parts = num_parts
        self.theta_max = theta_max

        self.worm = []
        for i in range(num_parts):
            self.worm.append(LineString([(0, i), (0, i + 1)]))

        self.food = Point(food_coord)
        self.disToFood = Point(self.worm[-1].coords[1]).distance(self.food)
        self.angles = np.zeros(num_parts)

    def step(self, theta):
        assert len(theta) == self.num_parts, "Number of outputs doesn't match the number of parts"

        rel_theta = np.zeros(self.num_parts)

        for i, t in enumerate(theta):
            rel_theta[i] = t - self.angles[i]
            self.angles[i] = t

        for i in range(self.num_parts):
            self.worm[i:] = affinity.rotate(MultiLineString(self.worm[i:]), rel_theta[i], origin=self.worm[i].coords[0])

        d = Point(self.worm[-1].coords[1]).distance(self.food)

        if d < self.disToFood:
            r = 1
        else:
            r = -1
        self.disToFood = d

        left_angle = self.theta_max - self.angles
        right_angle = self.angles + self.theta_max

        return r, left_angle, right_angle, d

    def render(self):
        xy = [l.xy for l in self.worm]

        x = [p[0][0] for p in xy]
        x.append(xy[-1][0][1])
        y = [p[1][0] for p in xy]
        y.append(xy[-1][1][1])

        ax = np.asarray(x)
        ay = np.asarray(y)

        plt.ion()
        plt.clf()
        plt.plot(ax, ay)
        plt.scatter(self.food.xy[0], self.food.xy[1])

        bound = self.num_parts + 2
        plt.axis([-bound, bound, -bound, bound])
        plt.pause(0.05)
