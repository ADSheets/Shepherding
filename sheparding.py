k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:06:09 2020

@author: asheets
"""

import numpy, cv2, time
from tqdm import tqdm
from random import gauss
from math import pi, sqrt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
import matplotlib.pyplot as plt

TWOPI = 2*pi

FLOCK_SIZE = 30
PASTURE_SIZE = 150
ONE = 1/PASTURE_SIZE
BUFFER = ONE / 10   #Minimum allowable distance from obstacle

rs = 65*ONE         #shepard detection distance
ra = 2*ONE          #agent to agent interaction distance
c = 1.05            #relative strength of attraction to the n nearest neighbours
rho_a = 2           #relative strength of repulsion from other agents
rho_s = 1           #relative strength of repulsion from the shepherd
h = .5              #relative strength of proceeding in the previous direction
e = .3              #relaive strength of angular noise
d = 2.5             #relative strength of repulsion from obstacles
MOVE_FREQ = .1      #frequency of moving while grazing

max_delta = 1*ONE   #agent displacement per time step
n = FLOCK_SIZE - 2  #number of nearest neighbours
    
N = FLOCK_SIZE      #number of agents
fn = lambda N : ra*N**(2/3)
max_delta_s = 5*ONE #Shepard displacement per time step
        
def unit(a):
    normed = a / norm(a, axis=1)[:,numpy.newaxis]
    return numpy.nan_to_num(normed)
        
class Flock:
    def __init__(self, N):
        self.e_hat = 0  
                
        self.N = N
        
        self.xy = numpy.random.random((N, 2)) * .05 + .8
        self.H_hat = numpy.zeros((N, 2))
        
        self.tree = cKDTree(self.xy)
    
    def step(self, S_bar, goal, obstacles):
        A_bar = self.xy

        #Agent-shepard repulsion
        R_bar_s = A_bar - S_bar
        
        neighboring = self.tree.query_ball_point(self.xy, ra)
        neighboring = [list( set(neighbors).difference(set([i])) ) for i,neighbors in enumerate(neighboring)]
        pairwise_distance = squareform(pdist(self.xy))
        numpy.fill_diagonal(pairwise_distance, float('inf'))
        
        C_bar = numpy.zeros((self.N, 2))
        R_bar_a = numpy.zeros((self.N, 2))
        for i, neighbors in enumerate(neighboring):
            if neighbors:
                Ai = self.xy[i]
                Aj = self.xy[neighbors]

                #Separation
                R_bar_a[i,:] = ( (Ai - Aj) / norm(Ai - Aj) ).sum(axis=0) 

                #Cohesion
                n_nearest = numpy.argsort(pairwise_distance[i,:])[:n]
                LCM = self.xy[n_nearest].mean(axis=0)
                C_bar[i,:] = LCM - A_bar[i,:]

        rs_mask = norm(A_bar - S_bar, axis=1) > rs        
        R_bar_s[rs_mask] = numpy.array([0, 0])
        C_bar[rs_mask] = numpy.array([0, 0])
        
        #Obstacle avoidance
        deflection = numpy.zeros((self.N, 2))
        if obstacles:
            for i in range(self.N):
                curbfeeler = CurbFeeler(self.xy[i,:].reshape((1,2)), ra*2, self.H_hat[i,:].reshape((1,2)))
                intersections = []
                for obstacle in obstacles:
                    if curbfeeler.intersects(obstacle):
                        point_of_intersection = curbfeeler.point_of_intersection_with(obstacle)
                        intersections.append(norm(self.xy[i,:] - point_of_intersection))
                    else:
                        intersections.append(float('inf'))
                closest_intersection = min(intersections) 
                if closest_intersection == float('inf'):
                    continue
                obstacle = obstacles[intersections.index(closest_intersection)]
                deflection[i,:] = curbfeeler.deflect(obstacle) / closest_intersection
            
        deflection = unit(deflection)
        C_hat = unit(C_bar)
        R_bar_a_hat = unit(R_bar_a)
        R_bar_s_hat = unit(R_bar_s)
        
        #angular noise
        random_theta = numpy.random.random((self.N,1))*TWOPI
        epsilon_hat = numpy.hstack((numpy.cos(random_theta), numpy.sin(random_theta)))
        
        H_bar_prime = h * self.H_hat + c * C_hat + rho_a * R_bar_a_hat + rho_s * R_bar_s_hat + e * epsilon_hat        
        H_hat_prime = unit(H_bar_prime)
        
        delta = numpy.zeros((self.N, 1))
        delta[numpy.logical_or(~rs_mask, numpy.random.random((self.N)) < MOVE_FREQ)] = max_delta

        #Prohibit passing through obstacles TODO: abstract with obstacle avoidance
        if obstacles:
            for i in range(self.N):
                curbfeeler = CurbFeeler(self.xy[i,:].reshape((1,2)), delta[i], H_hat_prime[i,:].reshape((1,2)))
                intersections = []
                for obstacle in obstacles:
                    if curbfeeler.intersects(obstacle):
                        point_of_intersection = curbfeeler.point_of_intersection_with(obstacle)
                        intersections.append(norm(self.xy[i,:] - point_of_intersection))
                    else:
                        intersections.append(float('inf'))
                closest_intersection = min(intersections) 
                if closest_intersection != float('inf'):
                    delta[i] = closest_intersection - BUFFER

        #Dont leave the goal zone 
        mask = goal.inside(self.xy)
        delta[mask.reshape(delta.shape)] = 0
        
        self.xy += delta*H_hat_prime
        self.H_hat = H_hat_prime
        self.tree = cKDTree(self.xy)
        
        return delta.mean()

class Shepard:
    def __init__(self):
        self.xy = numpy.array([0,0], dtype=numpy.float64).reshape(1,2)                
        self.p=numpy.array([-1,-1]).reshape(1,2)
    
    def move_toward(self, p, flock_xy, obstacles):
        if (norm(flock_xy - self.xy, axis=1) < 3 * ra).any():
            delta = .3*ra
        else:
            delta = max_delta_s
        
        H_prime = unit(p - self.xy)

        self.xy += H_prime * delta
        
    def step(self, flock_xy, goal, obstacles):
        #Ignore those in goal zone 
        goal_center = goal.center
        mask = goal.inside(flock_xy)
        flock_xy = flock_xy[~mask]
        
        GCM = flock_xy.mean(axis=0).reshape(1,2)
        distance_to_GCM = norm(flock_xy - GCM, axis=1)
        orientation = unit(flock_xy - GCM) - unit(goal.center - GCM)
        distance_to_GCM[(orientation < 1.42).all(axis=1)] = 0 #Don't collect agents between GCM and goal
        
        if (distance_to_GCM < fn(flock_xy.shape[0])).all():
            Pd = unit(GCM - goal_center) * ra * sqrt(N) + GCM
            self.move_toward(Pd, flock_xy, obstacles)
            self.p=Pd
        else:
            Af = flock_xy[numpy.argsort(distance_to_GCM)[-1], :]
            Pc = unit(Af - GCM) * ra + Af
            self.move_toward(Pc, flock_xy, obstacles)
            self.p=Pc
        
class Enviroment:
    def reset(self, goal, obstacles):
        self.goal = goal
        self.obstacles = obstacles
        self.num_step = 0

        self.shepard = Shepard()
        self.flock = Flock(FLOCK_SIZE)

        self.fig = None        
        
    def step(self):
        self.flock.step(self.shepard.xy, self.goal, self.obstacles)
        self.shepard.step(self.flock.xy, self.goal, self.obstacles)
        
    def render(self):
        if not self.fig:
            self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.axis('off')

        ax.scatter(self.flock.xy[:,0], self.flock.xy[:,1], color='r')
        ax.scatter(self.shepard.xy[:,0], self.shepard.xy[:,1], color='b')
        ax.scatter(self.shepard.p[:,0], self.shepard.p[:,1], color='g')

        ax.plot(self.goal.p1,self.goal.p2, 'k-')
        ax.plot(self.goal.p1,self.goal.p4, 'k-')
        ax.plot(self.goal.p3,self.goal.p2, 'k-')
        ax.plot(self.goal.p3,self.goal.p4, 'k-')
        
        ax.set_ylim(-.25,1.25)
        ax.set_xlim(-.25,1.25)

        self.fig.canvas.draw_idle()

        data = numpy.fromstring(self.fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        self.fig.clf()
        
        return data

class CurbFeeler:
    def __init__(self, xy=None, magnitude=None, heading=None, points=None):
        if not points:
            self.magnitude = magnitude
            self.heading = heading
            self.theta = numpy.arctan2(heading[0,1], heading[0,0])
            
            self.Ax, self.Ay = xy[0]
            self.Bx, self.By = (xy + magnitude * heading)[0]
            self.delta_x = self.Bx - self.Ax
            self.delta_y = self.By - self.Ay
        else:
            (self.Ax, self.Ay), (self.Bx, self.By) = points
            self.delta_x = self.Bx - self.Ax
            self.delta_y = self.By - self.Ay
            self.theta = numpy.arctan2(self.delta_y, self.delta_x)
            self.magnitude = sqrt( self.delta_x**2 + self.delta_y**2 )
            self.heading = numpy.array([numpy.cos(self.theta), numpy.sin(self.theta)])
            
    def intersects(self, other):
        if type(other) == Square: return other.intersects(self)
                        
        Ax, Ay, Bx, By = self.Ax, self.Ay, self.Bx, self.By
        Cx, Cy, Dx, Dy = other.Ax, other.Ay, other.Bx, other.By
        
        a1 = numpy.array([[Ax - Cx, Bx - Cx],
                          [Ay - Cy, By - Cy]])
        b1 = numpy.array([[Ax - Dx, Bx - Dx],
                          [Ay - Dy, By - Dy]])
        a1 = numpy.linalg.det(a1)
        b1 = numpy.linalg.det(b1)
        
        if (a1 < 0) == (b1 < 0):
            return False
        
        a2 = numpy.array([[Cx - Ax, Dx - Ax],
                          [Cy - Ay, Dy - Ay]])
        b2 = numpy.array([[Cx - Bx, Dx - Bx],
                          [Cy - By, Dy - By]])
    
        a2 = numpy.linalg.det(a2)
        b2 = numpy.linalg.det(b2)
        
        if (a2 < 0) == (b2 < 0):
            return False
        else:
            return True
    
    def deflect(self, other):
        trajectory1 = CurbFeeler(points=[(self.Ax, self.Ay),(other.Ax, other.Ay)])
        trajectory2 = CurbFeeler(points=[(self.Ax, self.Ay),(other.Bx, other.By)])
        theta1 = min(abs(trajectory1.theta - self.theta) % pi, abs(trajectory1.theta - pi - self.theta) % pi)
        theta2 = min(abs(trajectory2.theta - self.theta) % pi, abs(trajectory2.theta - pi - self.theta) % pi)
        if theta1 < theta2:
            trajectory = trajectory1
        else:
            trajectory = trajectory2
            
        return unit(trajectory.heading.reshape((1,2)))
    
    def point_of_intersection_with(self, other):
        xdiff = self.delta_x, other.delta_x
        ydiff = self.delta_y, other.delta_y
        
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')
    
        d = (det((self.Ax, self.Ay),(self.Bx, self.By)), det((other.Ax, other.Ay),(other.Bx, other.By)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        
        return numpy.array([-x, -y])
    
class Square:
    def __init__(self, p1, p2):
        self.x1, self.y1 = p1
        self.x2, self.y2 = p2
        
        self.xmin, self.xmax = sorted([self.x1, self.x2])
        self.ymin, self.ymax = sorted([self.y1, self.y2])

        self.p1 = (self.xmin, self.ymin)
        self.p2 = (self.xmin, self.ymax)
        self.p3 = (self.xmax, self.ymax)
        self.p4 = (self.xmax, self.ymin)
        
        self.x_center = (self.x1 + self.x2 / 2) + self.xmin
        self.y_center = (self.y1 + self.y2 / 2) + self.ymin
        
        self.center = numpy.array([self.x_center, self.y_center]).reshape((1,2))
        
    def inside(self, xy):
        mask1 = xy < numpy.array([self.xmax, self.ymax]).reshape((1,2))
        mask2 = xy > numpy.array([self.xmin, self.ymin]).reshape((1,2))
        
        return numpy.logical_and(mask1, mask2).all(axis=1)
        
    def intersects(self, other : CurbFeeler):
        for points in [(self.p1, self.p2),(self.p1, self.p4),(self.p3, self.p2),(self.p2, self.p4)]:
            if other.intersects(CurbFeeler(points=points)):
                return True
        return False
        
def main():    
    global env
    env = Enviroment()
    for i in range(3):
        env.reset(goal = Square((-.05, -.05), (.2,.2)), obstacles=[])
    
        out = cv2.VideoWriter(f'/home/asheets/herder {i}.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'), 60,
                              (432, 288))
    
        for _ in tqdm(range(8000)):
            data = env.render()
            out.write(data)
            env.step()
            
        out.release()
        plt.close(env.fig)
   
if __name__ == '__main__':
    main()