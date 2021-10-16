# -*- coding: utf-8 -*-
import random
import cv2 as cv

import scipy
import numpy as np

#import tensorflow as tf
import torch
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
from utils_torch import EnsurePath
from utils_torch.plot import *
from utils.model import *

def InitFromParam(param):
    return PlaceCells2D(param)

class PlaceCells2D(object):
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, ClassPath="Models.PlaceCells2D", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is not None:
            self.param = param
        else:
            param = self.param
            cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        self.CalculateXYs()
        self.SetXYs2ActivityMethod()
    def CalculateXYs(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            if HasAttrs(param.XYs, "Init"):
                EnsureAttrs(param.XYs.Init, "Method", default="FunctionCall")
                if param.XYs.Init.Method in ["FunctionCall"]:
                    utils_torch.CallFunctions(param.XYs.Init.Args, ObjCurrent=self, ObjRoot=utils.ArgsGlobal)
                else:
                    raise Exception()
    def SetXYs(self, XYs):
        param = self.param
        data = self.data
        cache = self.cache
        data.XYs = XYs
        #SetAttrs(param, "XYs", value="&data.XYs")
    def SetXYs2ActivityMethod(self):
        param = self.param
        data = self.data
        Functions = []
        methods = param.XYs2Activity.Init
        for method in methods:
            if method.Type in ["XYs2dLs"]:
                Function = lambda XYs:utils_torch.geometry2D.XYsPair2Distance(XYs, data.XYs)
            elif method.Type in ["DiffGaussian"]:
                GaussianCurve1 = utils_torch.math.GetGaussianCurveMethod(method.Amp1, method.Mean1, method.Std1)
                GaussianCurve2 = utils_torch.math.GetGaussianCurveMethod(method.Amp2, method.Mean2, method.Std2)
                Function = lambda dLs:GaussianCurve1(dLs) - GaussianCurve2(dLs)
            elif method.Type in ["Gaussian"]:
                GaussianCurve = utils_torch.math.GetGaussianCurveMethod(method.Amp, method.Mean, method.Std)
                Function = lambda dLs:GaussianCurve(dLs)
            elif method.Type in ["Norm2Mean0Std1"]:
                Function = lambda Activity:utils_torch.math.Norm2GivenMeanStd(Activity, 0.0, 1.0)
            elif method.Type in ["Norm2Sum1", "Norm2Probability"]:
                Function = utils_torch.math.Norm2Sum1
            elif method.Type in ["Norm2Min0"]:
                Function = utils_torch.math.Norm2Min0
            else:
                raise Exception(method)
            Functions.append(Function)
        self.XYs2Activity = utils_torch.StackFunction(*Functions)
    def GetActivityStatisticsInBoundaryBox(self, BoundaryBox, Resolution=50):
        ResolutionX, ResolutionY = utils_torch.plot.ParseResolution(BoundaryBox.Width, BoundaryBox.Height, Resolution)
        XYs = utils_torch.geometry2D.LatticeXYs(BoundaryBox, ResolutionX, ResolutionY, Flatten=True)
        activity = self.XYs2Activity(XYs)
        return utils_torch.math.NpArrayStatistics(activity)
    def XYs2ActivityDefault(self, XYs):
        # @param XYs: np.ndarray [PointNum, (x, y)]
        # @return Activity: np.ndarray [PointNum, PlaceCells.Num]
        raise Exception()
        return
    def PlotXYs(self, ax=None, Save=False, SavePath=None):
        if SavePath is None:
            SavePath=utils_torch.GetMainSaveDir() + "PlaceCells2D-XYs.png"
        data = self.data
        if ax is None:
            plt.close("all")
            fig, ax = plt.subplots()
        utils_torch.plot.PlotPoints(ax, data.XYs)
        if Save:
            utils_torch.EnsureFileDir(SavePath)
            plt.savefig(SavePath)
        return ax
    def bind_arenas(self, arenas, index=None):
        self.arenas = arenas
        if index is None:
            index = self.dict.setdefault('arena_index', 0)    
        self.arena = self.arenas.Getarena(index)
            
        if self.load:
            self.coords = self.dict['coords'].to(self.device)
            self.coords_np = self.coords.detach().cpu().numpy()
        else:
            self.coords_np = self.arena.Getrandom_xy(self.N_num) # [N_num, (x,y)]
            self.coords = self.dict['coords'] = torch.from_numpy(self.coords_np).to(self.device)
            '''
            x = torch.zeros((self.N_num)).to(device)
            torch.nn.init.uniform_(x, a=-self.box_width/2, b=self.box_width/2)
            y = torch.zeros((self.N_num)).to(device)
            torch.nn.init.uniform_(y, a=-self.box_height/2, b=self.box_height/2)
            self.coords = torch.stack([x, y], dim=1) #[pc_num, 2]
            self.dict['coords'] = self.coords
            '''
        self.xy = self.coords
        self.xy_np = self.coords_np
        self.type = self.dict['type']
        self.act_decay = search_dict(self.dict, ['act_decay', 'sigma'])
        self.act_decay_2 = self.act_decay ** 2
        self.act_center = search_dict(self.dict, ['act_center', 'peak'])
        self.norm_local = search_dict(self.dict, ['norm_local'], default=True, write_default=True)

        #print('PlaceCells: type:%s'%self.type)
        if self.type in ['diff_gaussian', 'diff_gauss']:
            self.GetAct = self.GetAct_dual_
            self.act_ratio = self.dict['act_ratio']
            self.act_positive = self.dict['act_positive']
            self.act_ratio_2 = self.act_ratio ** 2
            self.act_ratio_4 = self.act_ratio ** 4
            # minimum of difference gaussian curve is (ratio^4 ** (ratio^2/(1-ratio^4)) - 1/ratio^2 * ratio^4 ** (1/(1-ratio^4)))
            self.minimum = self.act_ratio_4 ** ( self.act_ratio_2 / (1 - self.act_ratio_2) ) - ( 1 / self.act_ratio_2 ) * ( self.act_ratio_4 ** ( 1 / (1 - self.act_ratio_2)) )
            self.separate_softmax = search_dict(self.dict, ['separate_softmax'], default=False, write_default=True)

            #print('act_positive:%s'%(str(self.act_positive)))
        else:
            self.GetAct = self.GetActivation = self.GetAct_single
        if self.verbose:
            print('Place_Cells: type:%s act_decay:%f act_center:%f norm_local:%s separate_softmax:%s'% \
                (self.type, self.act_decay, self.act_center, self.norm_local, self.separate_softmax))

    def GetAct_batch(self, points): # points: [batch_size, (x,y)]
        points = torch.unsqueeze(points, dim=1) # [batch_size, 1, (x, y)]
        pc_act = self.GetAct(points) # [batch_size, 1, pc_num]
        pc_act = torch.squeeze(pc_act, dim=1) # [batch_size, pc_num]
        #print('GetAct_batch: pc_act.size: %s'%str(pc_act.size())) # str(·) is necessary
        return pc_act
    def GetAct_single(self, points): # points: [batch_size, step_num, (x,y)]
        points = points.to(self.device)
        points_expand = torch.unsqueeze(points, dim=2) #points_expand:[batch_size, step_num, (x,y), 1]
        coords_expand = torch.unsqueeze(torch.unsqueeze(self.coords, dim=0), dim=0) #points_expand:[place_cells_num, (x,y )]
        vec = points_expand - coords_expand # [batch_size, step_num, pc_num, (x,y)]
        dist = torch.sum(vec ** 2, dim=3) # [batch_size, step_num, pc_num, 1]
        dist = torch.squeeze(dist)
        act = torch.exp(- dist / (2 * self.act_decay_2))
        if self.norm_local:
            act /= torch.sum(act, dim=2, keepdim=True)         
        return self.act_center * act # pos:[batch_size, step_num, act]
    def GetAct_dual_(self, points):
        points = points.to(self.device)
        points_expand = torch.unsqueeze(points, dim=2)
        coords_expand = torch.unsqueeze(torch.unsqueeze(self.coords, dim=0), dim=0)
        vec = points_expand - coords_expand # [batch_size, step_num, pc_num, (x,y)]
        dist = torch.sum(vec ** 2, dim=3, keepdim=True) # [batch_size, step_num, pc_num, 1]
        act_0 = torch.exp(- dist / (2 * self.act_decay_2))
        act_1 = torch.exp(- dist / (2 * self.act_decay_2 * self.act_ratio_2)) / self.act_ratio_2
        pc_act = self.act_center * (act_0 - act_1)
        pc_act = torch.squeeze(pc_act, dim=3) # [batch_size, step_num, pc_num]
        #print('GetAct_dual_: pc_act.size: %s'%str(pc_act.size()))
        return pc_act.float()

    def GetAct_dual(self, points): # pos:[batch_size, step_num, (x,y)]
        points = points.to(self.device)
        points_expand = torch.unsqueeze(points, dim=2)
        coords_expand = torch.unsqueeze(torch.unsqueeze(self.coords, dim=0), dim=0)
        vec = points_expand - coords_expand # [batch_size, step_num, pc_num, (x,y)]
        dist = torch.sum(vec ** 2, dim=3, keepdim=True) # [batch_size, step_num, pc_num, 1]
        
        if not self.norm_local:
            act_0 = torch.exp(- dist / (2 * self.act_decay_2))
            act_1 = torch.exp(- dist / (2 * self.act_decay_2 * self.act_ratio_2)) / self.act_ratio_2
            act = act_0 - act_1

            if self.act_positive:
                act -= self.minimum
                #act += 1.0e-8
                act += 1.00e-50
        else: # act assumed to be positive, otherwise cannot do normalization.
            if self.separate_softmax:
                act = F.softmax(- dist / (2 * self.act_decay_2 ), dim=2) - F.softmax(- dist / (2 * self.act_decay_2 * self.act_ratio_2), dim=2)
            else:
                act = torch.exp(- dist / (2 * self.act_decay_2) ) - torch.exp(- dist / (2 * self.act_decay_2 * self.act_ratio_2)) / self.act_ratio_2 # [batch_size, step_num, pc_num]
                act -= self.minimum # only suitable for separate_softmax=False
            #act += torch.abs(torch.min(act, dim=2, keepdim=True)[0])
            #act += 1.00e-50 #avoid log(0)
            #print('place_cells_act_min:%s'%torch.min(act))
            #act /= torch.sum(act, dim=2, keepdim=True)
        '''
        if not self.normalize:
            act = torch.exp(- dist / (2 * self.sigma * self.sigma)) - torch.exp(- dist / (2 * self.sigma * self.sigma * self.ratio * self.ratio)) / self.ratio #(batch_size, step_num, place_cells_num)
            act += torch.abs(torch.min(act, dim=2, keepdim=True)[0])
        else:
            act = F.softmax(- dist / (2 * self.sigma * self.sigma), dim=2)
            act -= F.softmax( - dist / (2 * self.sigma * self.sigma * self.ratio * self.ratio), dim=2) #softmax will automacially do normalization.
            act += torch.abs(torch.min(act, dim=2, keepdim=True)[0])
            act /= torch.sum(act, dim=2, keepdim=True)
        act = torch.squeeze(act)
        '''
        act = torch.squeeze(act, dim=3)
        act_scale = self.act_center * act.float() # [batch_size, step_num, pc_num]
        return F.softmax(act_scale, dim=2)

    def Getnearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.
        Args: 
            activation: Place cell activations of shape [batch_size, step_num, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, step_num, 2].
        '''
        _, idxs = tf.math.top_k(activation, k=k)
        pred_pos = tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
        return pred_pos
        
    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        us_np = self.us.numpy()
        pc_outputs = pc_outputs.numpy().reshape(-1, self.N_num)
        
        T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(us_np, pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        return pc
    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = pos.astype(np.float32)
        #Maybe specify dimensions here again?
        pc_outputs = self.GetActivation(pos)
        pc_outputs = tf.reshape(pc_outputs, (-1, self.cell_num))
        C = pc_outputs@tf.transpose(pc_outputs)
        Csquare = tf.reshape(C, (res,res,res,res))
        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)
        return Cmean
    
    def GetAct_map(self, arena, res=50):
        width, height = self.arena.width, self.arena.height
        res_x, res_y = Getres_xy(res, width, height)
        points_int = np.empty(shape=[res_x, res_y, 2], dtype=np.int) #coord
        for i in range(res_x):
            points_int[i, :, 1] = i # y_index
        for i in range(res_y):
            points_int[:, i, 0] = i # x_index

        #print(points_int)
        points_float = Getfloat_coords_np( points_int.reshape(res_x * res_y, 2), self.arena.xy_range, res_x, res_y ) # [res_x * res_y, 2]
        #print(points_float.reshape(res_x, res_y, 2))
        points_tensor = torch.unsqueeze(torch.from_numpy(points_float).to(self.device), axis=0) # [1, res_x * res_y, 2]
        arena_mask = arena.Getmask(res_x=res_x, res_y=res_y, points_grid=points_float)
        pc_act = self.GetAct(points_tensor)

        # [1, res_x * res_y, place_cells_num] -> [res_x, res_y, place_cells_num]
        pc_act = pc_act.detach().cpu().numpy().squeeze() 
        pc_act = pc_act.reshape((res_x, res_y, self.N_num))
        pc_act = np.transpose(pc_act, (2,0,1))

        '''
        arena_mask = np.ones((res_x, res_y)).astype(np.bool)
        #print(points_float.shape)
        points_out_of_region = arena.out_of_region(points_float, thres=0.0)
        if points_out_of_region.shape[0] > 0:
            #print(points_out_of_region)
            #print(points_out_of_region.shape)
            points_out_of_region = np.array([points_out_of_region//res_x, points_out_of_region%res_x], dtype=np.int)
            
            points_out_of_region = np.transpose(points_out_of_region, (1,0))
            
            #for i in range(points_out_of_region.shape[0]):
            #    arena_mask[points_out_of_region[i,0], points_out_of_region[i,1]] = 0.0
            arena_mask[points_out_of_region] = 0.0
        '''
        
        #print(arena_mask)
        #print(arena_mask.shape)
        return pc_act, arena_mask
    def plot_place_cells_coords(self, ax=None, arena=None, save=True, save_path='./', save_name='place_cells_coords.png'):
        arena = self.arenas.Getcurrent_arena() if arena is None else arena
        if ax is None:
            plt.close('all')
            fig, ax = plt.subplots()
        arena.plot_arena(ax, save=False)
        ax.scatter(self.coords_np[:,0], self.coords_np[:,1], marker='d', color=(0.0, 1.0, 0.0), edgecolors=(0.0,0.0,0.0), label='Start Positions') # marker='d' for diamond
        ax.set_title('Place Cells Positions')
        if save:
            EnsurePath(save_path)
            #cv.imwrite(save_path + save_name, imgs) # so that origin is in left-bottom corner.
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)
            plt.close()
    
    def plot_place_cells(self, act_map=None, arena=None, res=50, plot_num=100, col_num=15, save=True, save_path='./', save_name='place_cells_plot.png', cmap='jet'):
        arena = self.arena if arena is None else arena
        
        if act_map is None:
            act_map, arena_mask = self.GetAct_map(arena=arena, res=res) # [N_num, res_x, res_y]
        else:
            res_x, res_y = act_map.shape[1], act_map.shape[2]
            arena_mask = arena.Getmask(res_x=res_x, res_y=res_y)

        act_map = act_map[:, ::-1, :] # when plotting image, default origin is on top-left corner.
        act_max = np.max(act_map)
        act_min = np.min(act_map)
        #print('PlaceCells.plot_place_cells: act_min:%.2e act_max:%.2e'%(act_min, act_max))

        if plot_num < self.N_num:
            plot_index = np.sort(random.sample(range(self.N_num), plot_num)) # default order: ascending.
        else:
            plot_num = self.N_num
            plot_index = range(self.N_num)

        img_num = plot_num + 1
        row_num = ( plot_num + 1 ) // col_num
        if img_num % col_num > 0:
            row_num += 1

        #print('row_num:%d col_num:%d'%(row_num, col_num))
        fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(5*col_num, 5*row_num))
        
        act_map_norm = ( act_map - act_min ) / (act_max - act_min) # normalize to [0, 1]

        cmap_func = plt.cm.Getcmap(cmap)
        act_map_mapped = cmap_func(act_map_norm) # [N_num, res_x, res_y, (r,g,b,a)]
        for i in range(act_map_mapped.shape[0]):
            act_map_mapped[i,:,:,:] = cv.GaussianBlur(act_map_mapped[i,:,:,:], ksize=(3,3), sigmaX=1, sigmaY=1)
        
        arena_mask_white = (~arena_mask).astype(np.int)[:, :, np.newaxis] * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float)
        #print(np.sum(arena_mask.astype(np.int)))
        #print(~arena_mask)
        #print(arena_mask_white.shape)
        act_map_mapped = act_map_mapped * arena_mask.astype(np.int)[np.newaxis, :, :, np.newaxis] + arena_mask_white[np.newaxis, :, :, :]
        #arena_mask = arena_mask[np.newaxis, :, :]
        #for i in range(act_map_mapped.shape[0]):
        #    act_map_mapped[i, arena_mask] = (1.0, 1.0, 1.0, 1.0)

        for i in range(plot_num):
            row_index = (i+1) // col_num
            col_index = (i+1) % col_num
            N_index = plot_index[i]
            ax = axes[row_index, col_index]
            im = ax.imshow(act_map_mapped[N_index], extent=(arena.x0, arena.x1, arena.y0, arena.y1)) # extent: rescaling axis to arena size.
            ax.set_xticks(np.linspace(arena.x0, arena.x1, 5))
            ax.set_yticks(np.linspace(arena.y0, arena.y1, 5))
            ax.set_aspect(1)
            ax.set_title('Place Cells No.%d @ (%.2f, %.2f)'%(N_index, self.xy[N_index][0], self.xy[N_index][1]))
            #self.arena.plot_arena(ax, save=False)
            
        for i in range(plot_num + 1, row_num * col_num):
            row_index = i // col_num
            col_index = i % col_num
            ax = axes[row_index, col_index]
            ax.axis('off')
        
        ax = axes[0, 0]
        ax.axis('off')
        norm = mpl.colors.Normalize(vmin=act_min, vmax=act_max)
        ax_ = ax.inset_axes([0.0, 0.4, 1.0, 0.2]) # left, bottom, width, height. all are ratios to sub-canvas of ax.
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=ax_, 
            #pad=.05, # ?Fraction of original axes between colorbar and new image axes
            fraction=1.0, 
            ticks=np.linspace(act_min, act_max, num=5),
            aspect = 5, # ratio of colorbar height to width
            anchor=(0.5, 0.5), # coord of anchor point of colorbar
            panchor=(0.5, 0.5), # coord of colorbar's anchor point in parent ax.
            orientation='horizontal')
        cbar.set_label('Average fire rate', loc='center')

        ax.axis('off')
        
        if save:
            EnsurePath(save_path)
            #cv.imwrite(save_path + save_name, imgs) # so that origin is in left-bottom corner.
            plt.savefig(save_path + save_name)
            plt.close()
        

    def Getcoords_from_act(self, output, sample_num=3): # output: [batch_size, step_num, pc_num], np.ndarray
        sample_num = 3
        '''
        sample_num = int(self.N_num / 100)
        if sample_num < 3:
            sample_num = 3
        '''
        index_max = np.argpartition(output, -sample_num, axis=2)[:, :, -sample_num:] # [batch_size, step_num, sample_num], remaining elements are k largest, but unsorted.
        coords_max = np.zeros((output.shape[0], output.shape[1], sample_num, 2), dtype=np.float)
       
        for i in range(output.shape[0]): # batch_size
            for j in range(output.shape[1]): # step_num
                coords_max[i, j, :, :] = self.xy_np[index_max[i, j, :], :]
        
        #xy_pred = xy_pred.mean(axis=2) # [plot_num, step_num, (x, y)]

        return coords_max.mean(axis=2)
        
__MainClass__ = PlaceCells2D
utils_torch.model.SetMethodForModelClass(__MainClass__)