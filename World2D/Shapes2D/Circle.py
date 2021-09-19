
import numpy as np
import utils

import utils_torch
from utils_torch.attrs import CheckAttrs, EnsureAttrs, GetAttrs, SetAttrs, HasAttrs, MatchAttrs

import World2D.Shapes2D as Shapes2D

class Circle(Shapes2D.Shape2D):
    def __init__(self, param=None):
        super().__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self, param=None):
        if param is not None:
            self.param = param
        else:
            param = self.param
        CheckAttrs(param, "Type", value="Circle")
        EnsureAttrs(param, "Initialize.Method", default="CenterRadius")
        if param.Initialize.Method in ["CenterRadius"]:
            if not HasAttrs(param, "Center"):
                param.Center = GetAttrs(param.Initialize, "Center")
            if not HasAttrs(param, "Radius"):
                param.Radius = GetAttrs(param.Initialize, "Radius")
        else:
            raise Exception()
        SetAttrs(param, "Center.X", GetAttrs(param.Center)[0])
        SetAttrs(param, "Center.Y", GetAttrs(param.Center)[1])
        self.Center = np.array(GetAttrs(param.Center), dtype=np.float)

        self.CalculateBoundaryBox()
    def CalculateBoundaryBox(self):
        param = self.param
        # Calculate Boundary Box
        if not HasAttrs(param, "BoundaryBox"):
            XMin = param.Center.X - param.Radius
            YMax = param.Center.X + param.Radius
            YMin = param.Center.Y - param.Radius
            YMax = param.Center.Y + param.Radius
            SetAttrs(param, "BoundaryBox", [XMin, YMin, YMax, YMax])
            SetAttrs(param, "BoundaryBox.XMin", XMin)
            SetAttrs(param, "BoundaryBox.YMax", YMax)
            SetAttrs(param, "BoundaryBox.YMin", YMin)
            SetAttrs(param, "BoundaryBox.YMax", YMax)
            SetAttrs(param, "BoundaryBox.Width", YMax - XMin)
            SetAttrs(param, "BoundaryBox.Height", YMax - YMin)
            SetAttrs(param, "BoundaryBox.Size", max(param.BoundaryBox.Width, param.BoundaryBox.Height))
    def Distance2Center(self, Points):
        Vector2Center = self.Center[np.newaxis, :] - Points # [point_num, (x, y)]
        Distance2Center = np.linalg.norm(Vector2Center, axis=-1) # [point_num]
        return Distance2Center
    def Getdist_and_theta_from_center(self, xy): # xy: [point_num, 2]
        vec_from_center = xy - self.Center[np.newaxis, :] # [point_num, (x, y)]
        dist_from_center = np.linalg.norm(vec_from_center, ord=2, axis=-1) # [point_num]
        theta_from_center = np.arctan2(xy[:, 1], xy[:, 0])
        return dist_from_center, theta_from_center
    def IsOutside(self, XYs, MinDistance2Border=0.0): # Points:[PointNum, (x, y)]. This method is only valid for convex polygon.
        return self.Distance2Center(XYs) > (self.param.Radius + MinDistance2Border)
    def IsInside(self, XYs, MinDistance2Border=0.0):
        return self.Distance2Center(XYs) + MinDistance2Border < self.param.Radius
    def Vector2NearstBorder(self, Points):
        return        
    def avoid_border_circle(self, xy, theta, thres=None): # xy: [batch_size, (x, y)], theta: velocity direction.
        if thres is None or thres in ["default"]:
            thres = self.border_region_width
        dist_from_center, theta_from_center = self.Getdist_and_theta_from_center(xy) # [batch_size]

        #print(theta_from_center)
        theta_offset = theta - theta_from_center # [batch_size] 
        theta_offset = np.mod(theta_offset + np.pi, 2*np.pi) - np.pi # range:(-pi, pi)
        #print(theta_offset.shape)
       
        towards_wall = np.abs(theta_offset) < np.pi/2
        is_near_wall = (dist_from_center > (self.radius - thres)) * towards_wall # [batch_size]
        #print(is_near_wall.shape)
        #print(dist_from_center.shape)

        theta_adjust = np.zeros_like(theta) # zero arrays with the same shape as theta.
        theta_adjust[is_near_wall] = np.sign(theta_offset[is_near_wall]) * ( np.pi/2 - np.abs(theta_offset[is_near_wall]) ) # turn to direction parallel to wall

        return is_near_wall, theta_adjust

    def avoid_border_free(self, position, theta, arena_dict=None):
        is_near_wall = ( np.zeros_like(theta) > 1.0 )
        theta_adjust = np.zeros_like(theta)
        return is_near_wall, theta_adjust

    def avoid_border_square(self, position, hd, box_width=None, box_height=None, arena_dict=None, thres=0.0):
        x = position[:,0]
        y = position[:,1]
        dists = [self.x1 - x, self.y1 - y, x - self.x0, y - self.y0] #distance to all edges (right, top, left, down)
        d_wall = np.min(dists, axis=0) #distance to nearst wall
        angles = np.arange(4)*np.pi/2 
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi
        
        is_near_wall = (d_wall < thres) * (np.abs(a_wall) < np.pi/2)
        theta_adjust = np.zeros_like(hd) #zero arrays with the same shape of hd.
        theta_adjust[is_near_wall] = np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))
        return is_near_wall, theta_adjust
    def plot_border(self, img): # img: [H, W, C], np.uint8. 
        return # to be implemented   
    def plot_arena_plt(self, ax=None, save=True, save_path="./", save_name="arena_random_xy.png", line_width=2, color=(0,0,0)):
        if ax is None:
            figure, ax = plt.subplots()

        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)
        ax.set_xticks(np.linspace(self.x0, self.x1, 5))
        ax.set_yticks(np.linspace(self.y0, self.y1, 5))
        ax.set_aspect(1)

        circle = plt.Circle(self.Center, self.radius, color=(0.0, 0.0, 0.0), fill=False)        
        ax.add_patch(circle)
        if save:
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)

        return ax
    
    def plot_random_xy_plt(self, ax=None, save=True, save_path="./", save_name="arena_random_xy.png", num=100, color=(0.0,1.0,0.0), plot_arena=True, **kw):
        if ax is None:
            figure, ax = plt.subplots()
            if plot_arena:
                self.plot_arena_plt(ax, save=False)
            else:
                ax.set_xlim(self.x0, self.x1)
                ax.set_ylim(self.y0, self.y1)
                ax.set_aspect(1) # so that x and y has same unit length in image.
        points = self.Getrandom_xy(num=100)
        ax.scatter(points[:,0], points[:,1], marker="o", color=color, edgecolors=(0.0,0.0,0.0), label="Start Positions")
        plt.legend()

        if save:
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)
        return ax

    def ReportInfo(self):
        param = self.param
        utils.AddLog("Circle2D: ", end="")
        utils.AddLog("Center:(%.1f, %.1f)"%(param.Center.X, param.Center.Y))
        utils.AddLog("Radius:(%.3f)"%param.Radius)

__MainClass__ = Circle