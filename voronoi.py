import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
import math as math
matplotlib.use('gtk3agg')

points = np.array([[0.56308613, 0.25387432],
       [0.72620782, 0.86302279],
       [0.40419715, 0.43198382],
       [0.20983254, 0.86107708]])


def create_points(noise):
    mean = (0, 0)
    cov = [[noise, 0], [0, noise]]
    uncerts = np.random.multivariate_normal(mean, cov, (8, 8))
    points = []
    y_delta = np.sqrt(3)
    for y in range(8):
        for x in range(8):
            xp = x*2
            yp = y_delta*y

            if y > 0 and y < 7:
                yp += uncerts[x, y, 1]

            if x > 0 and x < 7:
                xp += uncerts[x, y, 0]

            if y % 2 == 0:
                points.append([xp, yp])
            elif x < 7:
                points.append([xp+1, yp])

    return np.array(points)

def plot(points):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    ((pfig, triangles, circles), (tricircles, voronoi, combined)) = axs
    
    pfig.set_title('Points')
    triangles.set_title('Triangles')
    circles.set_title('Circles')
    tricircles.set_title('Triangles with Circles')
    voronoi.set_title('Voronoi Diagram')
    combined.set_title('All')

    # plot the points
    plot_points(axs.flatten(), points)

    tri = plot_delaunay([triangles, tricircles, combined], points)

    plot_circles([circles, tricircles, combined], points, tri)

    plot_voronoi([voronoi, combined], points)
            
    for ax in axs.flat:
        ax.set_aspect('equal')
    plt.show()


def plot_points(axs, points):
    # plot the points
    for ax in axs:
        ax.plot(points[:, 0], points[:, 1], 'o')


def plot_delaunay(axs, points):
    tri = Delaunay(points)

    # plot the Delaunay triangulation
    for ax in axs:
        ax.triplot(points[:, 0], points[:, 1], tri.simplices)

    return tri


def plot_circles(axs, p, tri):
    # plot the circles around the triangles
    for s in tri.simplices:
        circle(axs, p[s])


def plot_voronoi(axs, points):
    vor = Voronoi(points)

    # plot the finite Voronoi diagram
    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 not in ridge:
            polygon = [vor.vertices[j] for j in ridge]
            xs, ys = zip(*polygon)
            for ax in axs:
                ax.plot(xs, ys, color='k')

    # plot the infinite Voronoi diagram
    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge:
            perp(axs, vor, i)


def perp(axs, vor, index):
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

    for ax in axs:
        ax.plot(xs, ys, color='k')


def circle(axs, triangle):
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
    
    for ax in axs:
        circle = plt.Circle((cx, cy), radius, fill=False, color='g')
        ax.add_patch(circle)


def circle_points(r, n):
    angle = 2*np.pi/n
    points = []
    for i in range(0, n):
        x = r*math.cos(i*angle)
        y = r*math.sin(i*angle)
        points.append([x, y])

    return points


def create_circle_points(num_circles, num_points, points_delta):
    points = [[0, 0]]
    for i in range(1, num_circles):
        points += circle_points(i, num_points + num_points*(i-1)*points_delta)

    return np.array(points)
