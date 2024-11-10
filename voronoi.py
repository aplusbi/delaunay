import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
matplotlib.use('gtk3agg')

points = np.array([[0.56308613, 0.25387432],
       [0.72620782, 0.86302279],
       [0.40419715, 0.43198382],
       [0.20983254, 0.86107708]])


def create_points():
    points = []
    y_delta = np.sqrt(3)
    for y in range(8):
        for x in range(8):
            if y % 2 == 0:
                points.append([x*2, y_delta*y])
            else:
                points.append([x*2 + 1, y_delta*y])

    return np.array(points)

def plot(points):
    # plot the points
    plt.plot(points[:, 0], points[:, 1], 'o')

    plot_delaunay(points)

    plot_voronoi(points)
            
    plt.axis('equal')
    plt.show()


def plot_delaunay(points):
    tri = Delaunay(points)

    # plot the Delaunay triangulation
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)

    # plot the circles around the triangles
    for s in tri.simplices:
        circle(points[s])


def plot_voronoi(points):
    vor = Voronoi(points)

    # plot the finite Voronoi diagram
    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 not in ridge:
            polygon = [vor.vertices[j] for j in ridge]
            xs, ys = zip(*polygon)
            plt.plot(xs, ys, color='k')

    # plot the infinite Voronoi diagram
    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge:
            perp(vor, i)


def perp(vor, index):
    """
    Code taken from scipy voronoi_plot_2d
    """
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    simplex = np.asarray(vor.ridge_vertices[index])
    pointidx = vor.ridge_points[index]

    i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
    t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]])  # normal

    midpoint = vor.points[pointidx].mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, n)) * n
    if (vor.furthest_site):
        direction = -direction
    aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
    far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

    xs = [vor.vertices[i, 0], far_point[0]]
    ys = [vor.vertices[i, 1], far_point[1]]
    plt.plot(xs, ys, color='k')


def circle(triangle):
    p1 = triangle[0]
    p2 = triangle[1]
    p3 = triangle[2]
 
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    circle = plt.Circle((cx, cy), radius, fill=False, color='g')
    plt.gca().add_patch(circle)

