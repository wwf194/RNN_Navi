# -*- coding: utf-8 -*-
#import tensorflow as tf

import os
import math
import random
import sys
import warnings

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2 as cv

import config_sys
config_sys.set_sys_path()

import utils
from utils import get_from_dict, get_items_from_dict, search_dict, get_name_args, ensure_path, contain, remove_suffix, get_ax
from utils import build_Optimizer
from utils_plot import get_res_xy, plot_polyline, get_int_coords_np, norm_and_map
import utils_train
from utils_train import set_train_info
import model
from model import get_tensor_info, get_tensor_stat


from anal_grid import get_score

from Models.Place_Cells import Place_Cells

class Agent(object):
    def __init__(self, dict_, load=False):
        self.dict = dict_
        self.task = self.dict['task']
        self.input_mode = self.dict['input_mode']
        self.stop_prob = self.dict.setdefault('stop_prob', 0.0)
        self.step_num = self.dict.setdefault('step_num', 100)
        self.plot_heat_map = self.plot_act_map

        self.load = load

        if self.task in ['pc', 'pc_coord']:
            self.place_cells = Place_Cells(dict_ = self.dict['place_cells'], load=self.load)

        # loss settings
        self.loss_dict = self.dict['loss']
        #print(self.loss_dict)
        #print(self.loss_dict['main'])
        self.main_loss = self.loss_dict['main']['type']
        self.main_coeff = self.loss_dict['main']['coeff']
        self.act_coeff = self.loss_dict['act']
        self.weight_coeff = self.loss_dict['weight']
        
        self.dynamic_weight_coeff = self.loss_dict['dynamic_weight_coeff']['enable']
        if self.dynamic_weight_coeff:
            self.alt_weight_coeff_count = 0
            self.weight_ratio = self.loss_dict['dynamic_weight_coeff']['ratio_to_main_loss']
            self.weight_ratio_min = self.loss_dict['dynamic_weight_coeff']['ratio_to_main_loss_min']
            self.weight_ratio_max = self.loss_dict['dynamic_weight_coeff']['ratio_to_main_loss_max']
            #self.weight_coeff_0 = self.loss_dict['weight_coeff_0'] = self.weight_coeff      

        self.perform_list = {'main':0.0, 'act':0.0, 'weight':0.0}
        self.init_perform()
        self.loss_dict = self.dict['loss']
        self.main_loss = self.loss_dict['main']['type']
        self.main_coeff = self.loss_dict['main']['coeff']
        self.act_coeff = self.loss_dict['act']
        self.weight_coeff = self.loss_dict['weight']

        '''
        if self.task in ['pc', 'pc_coord']:
            self.dict['pc_error'] = 0.0
        '''
        self.sample_count = 0
        self.batch_count = 0

        self.cache = {}
    def report_perform(self, prefix='', verbose=True):
        report = prefix
        for key in self.perform_list.keys():
            report += '%s:%.4e '%(key, self.perform_list[key]/self.batch_count)
        #report += '\n'
        if verbose:
            print(report)
        return verbose
    def init_perform(self):
        self.batch_count = 0
        self.sample_count = 0
        if self.main_loss in ['mse', 'MSE']:
            self.perform_list['output_error_ratio'] = 0.0
    def reset_perform(self):
        self.batch_count = 0
        self.sample_count = 0
        for key in self.perform_list.keys():
            self.perform_list[key] = 0.0
    def get_perform(self):
        return # to be implemented

    def prep_data(self, path):
        dxy, xy_init, xy = get_items_from_dict(self.prep_path(path), ['input', 'input_init', 'output'])
        #print('Agent.prep_data: xy_init.shape: %s'%str(xy_init.size()))
        if self.task in ['pc']:
            data = {
                'input': dxy,
                'input_init': self.place_cells.get_act_batch(xy_init),
                'output': self.place_cells.get_act(xy),
            }
        elif self.task in ['coord']:
            data = {
                'input': dxy,
                'input_init': xy_init,
                'output': xy
            }
        return data
    def train(self, dict_, report_in_batch=None, report_interval=None, save_path=None, save_name=None, verbose=True):
        optimizer = self.cache['optimizer'] = utils.build_Optimizer(dict_['optimizer'], params=self.model.get_train_param())
        epoch_start, epoch_num, epoch_end = utils_train.get_epoch_info(dict_)
        utils_train.set_train_info(dict_)
        if save_path is None:
            save_path = dict_['save_path']
        else:
            dict_['save_path'] = save_path
        ensure_path(save_path)
        save, save_interval, save_before_train, save_after_train = get_items_from_dict(dict_,
            ['save', 'save_interval', 'save_before_train', 'save_after_train'])
        anal, anal_interval, anal_before_train, anal_after_train = get_items_from_dict(dict_,
            ['anal', 'anal_interval', 'anal_before_train', 'anal_after_train'])

        batch_num = dict_['batch_num']
        batch_size = dict_['batch_size']
        
        if report_in_batch is None:
            if not hasattr(self, 'report_in_batch'):
                report_in_batch = True
            else:
                report_in_batch = self.report_in_batch
        if report_in_batch:
            if report_interval is None:
                if not hasattr(self, 'report_interval'):    
                    report_interval = int(batch_num / 40)
                else:
                    report_interval = self.report_interval
        if save_before_train:
            self.save(save_path, self.dict['name'] + '_beforeTrain')

        if anal_before_train:
            self.anal(dict_, title='beforeTrain', save_path=save_path + 'anal_beforeTrain/')

        optimizer.update_before_train()

        epoch_index = epoch_start
        #print('epoch_index:%d epoch_end:%d'%(epoch_index, epoch_end))
        
        # main loop
        while epoch_index <= epoch_end:
            print('epoch=%d/%d'%(epoch_index, epoch_end), end=' ')
            # train model
            self.reset_perform()
            for batch_index in range(batch_num):
                #self.train_once(batch_size)
                path = self.walk_random(num=batch_size)
                data = self.prep_data(path)
                #print(data.keys())
                data = self.cal_perform(data)
                #get_tensor_stat(data['output_pred'], name='output_pred', complete=False)
                #get_tensor_stat(data['act'], name='firing rate', complete=False)
                #print(data.keys())
                #optimizer.train(data)
                # expansion of optimizer.train(data)
                optimizer.optimizer.zero_grad()
                loss = get_items_from_dict(data, ['loss'])
                #loss = results['loss']
                #self.model.get_weight_info()
                loss.backward()
                optimizer.optimizer.step()
                #print('*********************************************')
                #self.model.get_weight_info()
                #optimizer.optimizer.zero_grad()
                #self.model.anal_gradient()
                #self.model.plot_act(data['act'][0], save_path='../anal/act_plot/', save_name='act_plot_%d:%d.png'%(epoch_index, batch_index))
                #self.plot_path(path=path, save_path='../anal/path_plot/', save_name='path_plot._%d:%d.png'%(epoch_index, batch_index), model=self.model, plot_num=2)
                if report_in_batch:
                    if batch_index % report_interval == 0:
                        print('batch=%d/%d' % (batch_index, batch_num))
                        self.report_perform()
                        self.reset_perform()
                        # analysis
                        #print('lr: %.3e'%optimizer.get_lr())
                        #self.model.get_weight_info()
                        self.model.plot_act(data['act'][0], save_path='../anal/act_plot/', save_name='act_plot_%d:%d.png'%(epoch_index, batch_index))
                        self.plot_path(path=path, save_path='../anal/path_plot/', save_name='path_plot._%d:%d.png'%(epoch_index, batch_index), \
                            model=self.model, plot_num=2)
                        get_tensor_stat(data['output_pred'], name='output_pred', complete=False)
                        get_tensor_stat(data['act'], name='firing rate', complete=False)
                        self.model.anal_weight_change()
                        self.model.get_weight_stat(complete=False)
                        self.model.anal_gradient()
                #batch_num += 1
            train_perform = self.report_perform(prefix='train: ')

            #print('save:%s save_interval:%d'%(self.save, self.save_interval))
            if save and epoch_index % save_interval == 0:
                print('saving_model at this epoch')
                self.save(save_path, self.dict['name'] + '_epoch=%d' % epoch_index)
            
            if anal and epoch_index % anal_interval == 1:
                self.anal(dict_, title='epoch=%s'%epoch_index, save_path=save_path + 'anal_epoch=%s/'%epoch_index)
            
            optimizer.update_after_epoch()
            epoch_index += 1
        if save_after_train:
            self.save(save_path, self.dict['name'] + '_epoch=afterTrain')
        if anal_after_train:
            self.anal(dict_, title='afterTrain', save_path=save_path + 'anal_afterTrain/')
    def forward(self, data):
        data.update(self.model.forward(data))
        return data
    def cal_perform(self, data):
        output_truth = get_items_from_dict(data, ['output'])
        output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
        data.update({
            'output': output_truth,
            'output_pred': output,
            'act': act,
        })
        data.update(self.cal_perform_(data))
        return data

    def cal_loss_main_mse(self, data):
        return
    def cal_loss_main_cel(self, data):
        output_pred, output_truth = get_items_from_dict(data, ['output_pred', 'output_truth'])
        loss_main = - self.main_coeff * torch.mean( F.log_softmax(output_pred, dim=2) * output_truth)
        return loss_main
    def cal_perform_(self, data):
        input_, output_truth = get_items_from_dict(data, ['input', 'output'])
        
        #output_truth = transforms.Normalize(0, 1)(output_truth)
        batch_size = input_.size(0)
        #output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
        output_pred, act = get_items_from_dict(data, ['output_pred', 'act'])
        #print('output_pred_max: %s'%torch.max(torch.abs(output_pred)))
        #print('output_pred_mean: %s'%torch.mean(torch.abs(output_pred)))
        #self.model.get_weight_stat(complete=False)
        # normalization
        output_pred_mean = torch.mean(output_pred, dim=(0,1,2), keepdim=True).detach()
        output_pred_std = torch.std(output_pred, dim=(0,1,2), keepdim=True).detach()
        #print('output_pred_std:%s'%output_pred_std)
        
        # normalization
        #output_pred = (output_pred - output_pred_mean) / output_pred_std
        
        self.cache['output_mean'] = output_pred_mean
        self.cache['output_std'] = output_pred_std
        self.cache['act_mean'] = torch.mean(torch.abs(act))
        #print('output_pred_example: %s'%output_pred[0, -5])
        #print('output_pred_abs_mean: %s'%self.cache['output_mean'])
        #print('output_pred_std: %s'%self.cache['output_std'])
        
        # calculate main loss
        if self.main_loss in ['mse', 'MSE']:
            loss_main = self.main_coeff * F.mse_loss(output_pred, output_truth, reduction='mean')
            self.perform_list['output_error_ratio'] += ( torch.sum(torch.abs(output_pred - output_truth)) / torch.sum(torch.abs(output_truth)) ).item() # relative place cells prediction error
        elif self.main_loss in ['cel', 'CEL']:
            # output_pred: [batch_size, step_num, output_num]
            output_pred_log_softmax = F.log_softmax(output_pred, dim=2)
            output_truth_softmax = F.softmax(8.0 * output_truth, dim=2)
            #print(output_truth_softmax[-1, -1])
            #get_tensor_stat(output_truth_softmax, name='output_truth_softmax')
            
            output_pred_softmax = F.softmax(8.0 * output_pred, dim=2)
            #print(output_pred_softmax[-1, -1])
            #get_tensor_stat(output_pred_softmax, name='output_pred_softmax')

            #print(output_pred_softmax.size())
            #print('output_pred_softmax: %s' % output_pred_softmax[0,-5, 0:50])
            #print('output_truth_sum_dim2:%s'%torch.sum(output_truth, dim=2)) # should all be 1.000
            
            #print(torch.sum(output_pred_softmax, dim=2))
            #input()
            # [batch_size, step_num, output_num]
            loss_main = - self.main_coeff * torch.sum(torch.mean(output_pred_log_softmax * output_truth_softmax, dim=(0, 1)))
        else:
            raise Exception('Invalid main loss: %s'%str(self.main_loss))
        
        # calculate act loss
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        # calculate weight loss
        loss_weight = self.model.cal_loss_weight(coeff=self.weight_coeff)
        loss_weight = self.cal_loss_weight(loss_weight, loss_main)
        self.perform_list['weight'] += loss_weight.item()

        self.perform_list['act'] += loss_act.item()
        self.perform_list['main'] += loss_main.item()
        
        self.batch_count += 1
        self.sample_count += batch_size
        result = {
            'loss_main': loss_main,
            'loss_act': loss_act,
            'loss_weight': loss_weight,
            'loss': loss_main + loss_act + loss_weight
        }
        #print('loss_main:%.3e loss_act:%.3e loss_weight:%.3e'%(loss_main.item(), loss_act.item(), loss_weight.item()))
        #self.model.anal_weight_change()
        #self.model.anal_gradient()
        #input()
        return result
    def cal_loss_weight(self, loss_weight, loss_main):
        if self.weight_coeff > 0.0 and self.dynamic_weight_coeff:
                loss_ratio = loss_weight.item() / loss_main.item() # ratio of weight loss to pc loss.
                #print('loss_ratio:%.3f'%loss_ratio)
                if self.weight_ratio_min < loss_ratio < self.weight_ratio_max:
                    loss_weight = loss_weight
                else:
                    weight_coeff_0 = self.weight_coeff
                    self.weight_coeff = self.weight_coeff * self.weight_ratio / loss_ratio # alternating weight cons index so that loss_weight == loss_main * dynamic_weight_ratio
                    self.alt_weight_coeff_count += 1
                    if self.alt_weight_coeff_count > 50:
                        print('alternating weight_coeff from %.3e to %.3e'%(weight_coeff_0, self.weight_coeff))  
                        self.alt_weight_coeff_count = 0
                    loss_weight = self.weight_coeff / weight_coeff_0 * loss_weight # alter loss_weight scale
        else:
            loss_weight = self.weight_coeff * loss_weight
        return loss_weight
    def cal_perform_pc_coord(self, data):
        #x, y = self.prep_path(path)
        y = get_items_from_dict(data, ['output'])
        batch_size = y.size(0)
        output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.get_act(y)
        loss_coord =self.main_coeff_pc * F.mse_loss(output[:, :, 0:2], pc_output, reduction='mean')
        loss_main = self.main_coeff_coords * F.mse_loss(output[:, :, 2:-1], pc_output, reduction='mean')
        self.perform_list['pc_error_ratio'] += ( torch.sum(torch.abs(output[:, :, 2:-1] - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight = self.weight_coeff * ( torch.mean(self.get_o() ** 2) + torch.mean(self.get_i() ** 2) )
        
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['coord'] += loss_coord.item()
        self.perform_list['pc'] += loss_main.item()
        
        self.batch_count += 1
        self.sample_count += batch_size
        return {
            'loss_main': loss_main,
            'loss_coord': loss_coord,
            'loss_act': loss_act,
            'loss_weight': loss_weight,
            'loss': loss_main + loss_coord + loss_act + loss_weight
        }
    '''
    def receive_options(self, options):
        self.options = options
        self.device = self.options.device
        self.arenas = self.options.arenas
        self.model = self.options.model if hasattr(options, 'model') else None
    '''
    def get_init_xy(self, init_method, arena, batch_size):
        init_method, init_args = get_name_args(init_method)
        
        if init_method in ['uniform', 'random']:
            return arena.get_random_xy(batch_size)

    def prep_path(self, path): # prep data from path to feed to rnn model.
        if self.input_mode in ['v_xy']:
            i = torch.from_numpy(path['dx_dy']).float() # [batch_size, step_num, (v_x, v_y)]
            o = torch.from_numpy(path['xy']).float() # [batch_size, step_num, (x, y)]
        elif self.input_mode in ['v_hd']:
            i = torch.from_numpy(np.stack((path['theta_xy'][:,:,0], path['theta_xy'][:,:,1], path['delta_xy']), axis=-1)).float() # [batch_size, step_num, (cos, sin, v)]
            o = torch.from_numpy(path['xy']).float() # [batch_size, step_num, (x, y)]
        else:
            raise Exception('Unknown input mode:'+str(self.input_mode))
        i_init = torch.from_numpy(path['xy_init']).float() # [batch_size, 2]
        #i = inputs.to(self.device)
        #i_iniy = init.to(self.device)
        #o = outputs.to(self.device)
        return {
            'input': i,
            'input_init': i_init,
            'output': o,
        }
        # return (inputs, init), outputs
    def walk_random(self, step_num=None, **kw): # step_num, t_total, random_init=False, arena_index=None, use_test_arenas_=None, full_info=False): #return random trajectories
        step_num = self.step_num if step_num is None else step_num
        batch_size = search_dict(kw, ['batch_size', 'num'], default=100)
        #step_num = search_dict(kw, ['t_total', 'sequence_length', 'step_num'], default=self.step_num)
        arena = search_dict(kw, ['arena'], default=self.arenas.get_current_arena())
        init_method = search_dict(kw, 'init_method', default='uniform')

        items = kw.setdefault('items', None)
        if items is None:
            mode = kw.setdefault('mode', 'train')
        else:
            mode = 'self-defined'
        
        dt = 0.02  # time step increment (seconds)
        xy = np.zeros([batch_size, step_num + 1, 2]) # [batch_num, step_num + 2, (x,y)]
        xy[:, 0, :] = self.get_init_xy(init_method, arena, batch_size)
        dl = np.zeros([batch_size, step_num]) # dl = v * dt : [batch_size, step_num]
        
        theta = np.zeros([batch_size, step_num + 1]) # head direction: [batch_num, step_num + 1]
        theta[:,0] = np.random.uniform(0, 2*np.pi, batch_size)
        
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        mu = 0  # turn angle bias 
        d_theta = np.random.normal(mu, sigma, [batch_size, step_num]) * dt # random angular velocity.
        
        b = 0.13 * 2 * np.pi # forward velocity rayleigh dist scale (m/sec)
        v_random = np.random.rayleigh(b, [batch_size, step_num]) # random velocity.
        
        if self.stop_prob > 0.0:
            offset = - math.log(1.0 - self.stop_prob, math.e) * 2 * (b ** 2) # CDF of rayleigh distribution is 1 - exp{ - x^2 / (2 * b^2) }
            v_random = v_random - offset
            v_random = ( 1 * (v_random > 0.0) ) * v_random
        '''
        count = 0
        zero_count = 0
        for i in range(random_vel.shape[0]):
            for j in range(random_vel.shape[1]):
                if(random_vel[i][j] < 0.0):
                    random_vel[i][j] = 0.0
                    zero_count += 1
                count += 1
        '''
        '''
        print(random_vel)
        print('v zero prob:%.5f'%(zero_count/count))
        input()
        # v = np.abs(np.random.normal(0, b*np.pi/2, [batch_size])) #[batch_size]
        '''
        
        for t in range(step_num):
            v = v_random[:,t]
            #theta_adjust = np.zeros(batch_size) #initialize theta_adjust
            d_theta_now = d_theta[:,t]

            is_near_wall, theta_adjust = arena.avoid_border(xy[:,t], theta[:,t])
            v[is_near_wall] *= 0.25 #slow down batches where the agent is near wall.

            d_theta_now += theta_adjust
            dl[:,t] = v * dt

            dxy = dl[:, t, None] * np.stack([np.cos(theta[:,t]), np.sin(theta[:,t])], axis=1) #[batch_size, 1] * [batch_size, 2] = [batch_size, 2]

            xy[:,t+1] = xy[:,t] + dxy #[batch_size, 1, 2]
            theta[:,t+1] = theta[:,t] + d_theta_now # Rotate head direction

        theta = np.mod(theta + np.pi, 2*np.pi) - np.pi # make sure head_dir in range [-pi, pi].
        theta_delta = np.diff(theta, axis=1) # [batch_num, step_num]

        path = {}
        sig = False

        path['dl'] = dl
        path['xy'] = xy[:,1:,:] # [batch_size, step_num, 2]
        if mode in ['train', 'full']:
            path['theta_xy'] = np.stack( [np.cos(theta), np.sin(theta)], axis=1 )[:,:-1,:] # head direction.
            path['theta_init'] = theta[:,0]
            path['xy_init'] = xy[:,0,:]
            path['dx_dy'] = np.diff(xy, axis=1) # [batch_size, step_num, 2]. position difference between current position and previous position.
            sig = True
        if mode in ['plot', 'full']:
            pass
            sig = True
        if mode in ['self-defined']:
            if contain(items, ['hd_delta_xy', 'theta_delta_xy']):
                path['theta_delta_xy'] = np.stack( [ np.cos(d_theta), np.sin(d_theta) ], axis=1 )[:,:-1] #head_dir difference between current position and previous position.
            sig = True
        if not sig:
            raise Exception('Agent.random_walk: invalid mode:%s'%str(mode))

        return path

    def plot_walk_random(self, arena=None, save=False, save_path='./', save_name='walk_random.png', plot_num=10, cmap='jet', **kw):
        arena = self.arenas.get_current_arena() if arena is None else arena

        path = self.walk_random(num=plot_num, arena=arena, mode='plot', **kw)
        
        #figure, ax = plt.subplots()
        figure, ax = plt.subplots(1, 1, figsize=(5*1.5, 5*1.0)) # figsize (width, length) in inches.
        
        ax.set_xlim(arena.x0, arena.x1)
        ax.set_ylim(arena.y0, arena.y1)
        ax.set_aspect(1) # so that x and y has same unit length in image.

        arena.plot_arena_plt(ax, save=False)

        #colors = get_colors(plot_num)

        dl = path['dl']
        dl_max = np.max(dl)
        dl_min = np.min(dl)
        if dl_max==dl_min:
            dl_norm = np.zeros(dl.shape, dtype=dl.dtype)
            dl_norm[:,:,:] = 0.5
        else:
            dl_norm = (dl - dl_min) / (dl_max - dl_min)
        
        cmap_func = plt.cm.get_cmap(cmap)
        dl_mapped = cmap_func(dl_norm) # [batch_size, ], res_x, res_y, (r,g,b,a)]        

        '''
        color = {
            'method':'start-end',
            'start':(0.0,0.0,1.0),
            'end':(0.0,1.0,0.0),
        }
        start = ax.plot([],[], c=color['start'], label='start')
        end = ax.plot([],[], c=color['end'], label='end')
        ax.legend(loc='upper right')
        '''

        for i in range(plot_num):
            color = {
                'method':'given',
                'data':dl_mapped[i]
            }
            plot_polyline(ax, path['xy'][i,:,:], color=color, width=2)
        
        ax_ = ax.inset_axes([1.05, 0.0, 0.12, 0.8])
        norm = mpl.colors.Normalize(vmin=dl_min, vmax=dl_max)
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=ax_, # occupy the ax exclusively to draw colorbar
            #ax=ax, # steal some space from ax to draw colorbar
            #pad=.05, # ?Fraction of original axes between colorbar and new image axes
            #fraction=1.0, # ratio to zoom in or out the colorbar
            ticks=np.linspace(dl_min, dl_max, num=5),
            #aspect = 10, # ratio of colorbar height to width
            #orientation='vertical',
            #location='left' # invalid key word
            )
        cbar.set_label('Step length')


        color_truth = (0.0, 1.0, 0.0)
        color_edge = (0.0, 0.0, 0.0)
        ax.scatter(path['xy'][:,0,0], path['xy'][:,0,1], marker='^', color=color_truth, edgecolors=color_edge, label='Start positions')
        ax.scatter(path['xy'][:,-1,0], path['xy'][:,-1,1], marker='o', color=color_truth, edgecolors=color_edge, label='End positions')
        ax.legend(bbox_to_anchor=(1.05, 0.7, 0.2, 0.3)) # (x, y, width, height). (x, y) is coordinate of lower left point.
        

        #start = ax.plot([1.0,2.0],[1.0,2.0], c=color['start'], label='start')
        #end = ax.plot([1.0,2.0],[1.0,2.0], c=color['end'], label='end')

        ax.set_title('%d random paths'%plot_num)
        
        plt.tight_layout()

        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)

        return ax
    def plot_path(self, path=None, model=None, plot_num=2, arena=None, save=True, save_path='./', save_name='path_plot', cmap='jet', **kw):
        if arena is None:
            arena = self.arenas.get_current_arena()
        # generate path
        if path is None:
            path = self.walk_random(num=plot_num, arena=arena, mode='full', **kw)
        else:
            #print(path.keys())
            batch_size = path['xy'].shape[0]
            if batch_size > plot_num:
                plot_index = random.sample(range(batch_size), plot_num)
                for key in path.keys():
                    path[key] = path[key][plot_index]
            else:
                plot_num = path.size(0)
        # get model prediction
        model = self.model if model is None else model
        if model is None:
            raise Exception('Agent.plot_heat_map: Error: model is None.')

        data = self.prep_data(path)
        #print(data.keys())
        data = self.cal_perform(data)
        output_pred = get_items_from_dict(data, ['output_pred']) # act: [plot_num, step_num, N_num]
        output_pred = output_pred.detach().cpu().numpy()
        if self.task in ['coord']:
            xy_pred = output_pred # [plot_num, step_num, (x, y)]
        elif self.task in ['pc']:
            xy_pred = self.place_cells.get_coords_from_act(output_pred) # [plot_num, step_num, sample_num, (x, y)]
            #print(xy_pred.shape)
            
            #print(xy_pred.shape)
        #print(path['xy'].shape)

        plt.close('all')
        figure, ax = plt.subplots(1, 1, figsize=(5*1.5, 5*1.0)) # figsize (width, length) in inches.
        ax.set_xlim(arena.x0, arena.x1)
        ax.set_ylim(arena.y0, arena.y1)
        ax.set_aspect(1) # so that x and y has same unit length in image.

        arena.plot_arena_plt(ax, save=False)

        dl = path['dl']
        '''
        dl_max = np.max(dl)
        dl_min = np.min(dl)
        if dl_max==dl_min:
            dl_norm = np.zeros(dl.shape, dtype=dl.dtype)
            dl_norm[:,:,:] = 0.5
        else:
            dl_norm = (dl - dl_min) / (dl_max - dl_min)
        
        cmap_func = plt.cm.get_cmap(cmap)
        dl_mapped = cmap_func(dl_norm) # [batch_size, res_x, res_y, (r,g,b,a)]
        '''
        #dl_mapped = norm_and_map(dl, cmap=cmap, return_min_max=True)
        dl_pred = np.diff(xy_pred, axis=2) # [batch_size, step_num, (x, y)]
        dl_pred = np.linalg.norm(dl_pred, ord=2, axis=2) # [batch_size, step_num]
        
        dl_cat_mapped, dl_min, dl_max = norm_and_map(np.concatenate([dl, dl_pred], axis=0), cmap=cmap, return_min_max=True)
        
        #dl_mapped = dl_cat_mapped[:, 0:dl.shape[1], :]
        #dl_pred_mapped = dl_cat_mapped[:, dl.shape[1]:, :]
        dl_mapped = dl_cat_mapped[0:dl.shape[0], :, :]
        dl_pred_mapped = dl_cat_mapped[dl.shape[0]:, :, :]

        color_truth = (0.0, 1.0, 0.0)
        color_pred = (0.0, 0.0, 1.0)
        color_edge = (0.0, 0.0, 0.0)

        for i in range(plot_num):
            color = {
                'method':'given',
                'data':dl_mapped[i]
            }
            plot_polyline(ax, path['xy'][i,:,:], color=color, width=1)
        for i in range(plot_num):
            '''
            color = {
                'method': 'start-end',
                'start': (1.0, 0.0, 0.0),
                'end': (0.0, 0.0, 1.0),
            }
            '''
            color = {
                'method': 'given',
                'data': dl_pred_mapped[i]
            }
            #print(xy_pred[i,:,:].transpose((1,0)))
            plot_polyline(ax, xy_pred[i,:,:], color=color, width=1)
            #input()

        ax.scatter(path['xy'][:,0,0], path['xy'][:,0,1], marker='^', color=color_truth, edgecolors=color_edge, label='Start positions')
        ax.scatter(path['xy'][:,-1,0], path['xy'][:,-1,1], marker='o', color=color_truth, edgecolors=color_edge, label='End positions')
        
        ax.scatter(xy_pred[:,0,0], xy_pred[:,0,1], marker='^', color=color_pred, edgecolors=color_edge, label='Start positions(Predicted)')
        ax.scatter(xy_pred[:,-1,0], xy_pred[:,-1,1], marker='o', color=color_pred, edgecolors=color_edge, label='End positions(Predicted)')
        #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0)) # coordinate of upper left point(ratio)
        ax.legend(bbox_to_anchor=(1.05, 0.7, 0.4, 0.3)) # (x, y, width, height). (x, y) is coordinate of lower left point.

        ax_ = ax.inset_axes([1.05, 0.0, 0.1, 0.6])
        norm = mpl.colors.Normalize(vmin=dl_min, vmax=dl_max)
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=ax_, # occupy the ax exclusively to draw colorbar
            #ax=ax, # steal some space from ax to draw colorbar
            #pad=.05, # ?Fraction of original axes between colorbar and new image axes
            #fraction=1.0, # ratio to zoom in or out the colorbar
            ticks=np.linspace(dl_min, dl_max, num=5),
            #aspect = 10, # ratio of colorbar height to width
            #orientation='vertical',
            #location='left' # invalid key word
            )
        cbar.set_label('Step length', loc='center') # loc is invalid parameter in matplotlib 3.1.3, but valid in 3.3.4. label is added along the long axis.
        #cbar.set_label('Step length')
        #start = ax.plot([1.0,2.0],[1.0,2.0], c=color['start'], label='start')
        #end = ax.plot([1.0,2.0],[1.0,2.0], c=color['end'], label='end')

        ax.set_title('%d random paths'%plot_num)
        
        plt.tight_layout() # try to make elements lie inside the canvas
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
        return ax
    def cal_act_map(self, model, arena, res=50, batch_size=200, batch_num=20, step_num=None):
        # Compute spatial firing fields
        N_num = model.dict['N_num']
        step_num = self.step_num if step_num is None else step_num

        res_x, res_y = get_res_xy(res, arena.width, arena.height)

        point_act = np.zeros([N_num, res_x, res_y])
        point_count  = np.zeros([res_x, res_y])

        #print('Agent.act_act_map: batch_num:%d batch_size:%d step_num:%d'%(batch_num, batch_size, step_num))
        out_count = 0
        point_num = batch_num * batch_size * step_num
        for index in range(batch_num):
            #print('batch_num:%d/%d'%(index, batch_num))
            with torch.no_grad():
                model.eval()
                path = self.walk_random(num=batch_size, arena=arena, step_num=step_num)
                output, act = get_items_from_dict(model.forward(self.prep_data(path)), ['output', 'act']) # act: [batch_size, step_num, N_num]
                #print(act[0])
                #input()
                xy_float = path['xy'] # [batch_size, step_num, 2]
                act = list(map(lambda x:x.detach().cpu(), act))

            #print(xy_float.shape)
            xy_int = get_int_coords_np(xy_float.reshape(batch_size * step_num, 2), arena.xy_range, res_x, res_y) # [batch_size, step_num, 2]
            #print(xy_int.shape)
            xy_int = xy_int.reshape(batch_size, step_num, 2) # [batch_size, step_num, 2]

            
            for i in range(batch_size): # batch_index
                for j in range(step_num): # time_step
                    x_int = xy_int[i, j][0]
                    y_int = xy_int[i, j][1]
                    #for k in range(N_num): #N_index
                    #try:
                    if 0<=x_int<res_x and 0<=y_int<res_y:
                        point_count[x_int, y_int] += 1.0
                        point_act[:, x_int, y_int] += act[i][j].numpy()
                    else:
                        out_count += 1
                    #except Exception:
                    #    print('%d %d %d %d'%(x_int, res_x, y_int, res_y))
        if out_count > 0:
            warnings.warn('Agent.cal_act_map: %.2f%%(%d/%d) points are out of arena.'%(out_count / point_num, out_count, point_num))

        for x in range(res_x):
            for y in range(res_y):
                if point_count[x, y] > 0.0:
                    point_act[:, x, y] /= point_count[x, y]
        
        '''
        if use_masks:
            masks = np.uint8(activations!=0.0)
        else:
            masks = None
        '''
        #print(point_count[10,10])

        return {
            'point_count': point_count,
            'point_act': point_act,
            'act_map': point_act #, masks
        }
    def get_relative_fire_rate(self, act_map, thres_up, thres_down):
        act_map_norm = (act_map - thres_down) / (thres_up - thres_down)
        act_map_norm[np.argwhere(act_map_norm>1.0)] = 1.0
        act_map_norm[np.argwhere(act_map_norm<0.0)] = 0.0
        return act_map_norm
    
    def anal_act(self, save_path='./', model=None, batch_size=None, batch_num=None, arena=None, separate_ei=None, act_map_res=50):
        model = self.model if model is None else model
        if model is None:
            raise Exception('Agent.anal_act: Error: model is None.')
        if separate_ei is None:
            separate_ei = model.separate_ei
        if arena is None:
            arena = self.arenas.get_current_arena()
        
        if batch_num is None:
            batch_num = int(batch_num / 10)
        act_map_info = self.cal_act_map(model=model, arena=arena, res=act_map_res, batch_size=batch_size, batch_num=batch_num) # [N_num, res_x, res_y]
        
        act_map, point_count = act_map_info['act_map'], act_map_info['point_count']

        self.plot_sample_num(point_count, arena, save_path=save_path)
        if self.task in ['pc']:
            self.plot_place_cells_prediction(model=model, act_map_info=act_map_info, save_path=save_path)
        self.plot_act_map(save=True, save_path=save_path, plot_num='all', model=model, batch_size=batch_size, batch_num=batch_num,  
            arena=arena, separate_ei=separate_ei, act_map_info=act_map_info)

    def plot_act_map(self, save=True, save_path='./', save_name='act_map.png', plot_num=None, model=None, act_map_info=None, batch_size=None, batch_num=None,
            arena=None, res=50, col_num=15, num_per_page=100, separate_ei=None, cmap='jet', sample_method='grid_score', map_method='individual'):

        model = self.model if model is None else model
        if model is None:
            raise Exception('Agent.plot_act_map: Error: model is None.')
        if arena is None:
            arena = self.arenas.get_current_arena()
    
        if separate_ei is None:
            separate_ei = model.separate_ei

        if act_map_info is None:
            #batch_size = trainer.batch_size
            if batch_num is None:
                batch_num = int(batch_num / 100)
            act_map_info = self.cal_act_map(model=model, trainer=trainer, arena=arena, res=res, batch_size=batch_size, batch_num=batch_num) 
                
        act_map, point_count = act_map_info['act_map'], act_map_info['point_count'] # [N_num, res_x, res_y]

        grid_score = get_score(act_map[:, :, :], coord_range=((arena.x0, arena.x1), (arena.y0, arena.y1)), save_path='./anal/')
        grid_score = np.array(grid_score[0]) # grid_score_60
        #grid_score_rank = np.argsort(-grid_score)
        
        act_map = act_map[:, :, ::-1] # when plotting image, default origin is on top-left corner.
        
        act_max = np.max(act_map)
        act_min = np.min(act_map)
        act_std = np.std(act_map)
        act_mean = np.mean(act_map)
        
        act_max_N = np.max(act_map, axis=(1, 2))
        act_min_N = np.min(act_map, axis=(1, 2))
        act_mean_N = np.mean(act_map, axis=(1, 2)) 
        act_std_N = np.std(act_map, axis=(1, 2))
        print('%s out of %s neurons have 0 act_std.'%(np.sum(act_std_N==0.0), model.N_num))

        act_thres_down = act_mean - 5 * act_std
        if act_thres_down < act_min:
            act_thres_down = act_min
        act_thres_up = act_mean + 5 * act_std
        if act_thres_up > act_max:
            act_thres_up = act_max

        act_info = {
            'act_max': act_max,
            'act_min': act_min,
            'act_std': act_std,
            'act_max_N': act_max_N,
            'act_min_N': act_min_N,
            'act_mean_N': act_mean_N,
            'act_std_N': act_std_N,
            'act_thres_up': act_thres_up,
            'act_thres_down': act_thres_down,
        }

        if map_method in ['universal', 'whole']: # normalize firing rate of all neurons to [0, 1]
            print('act_min:%.2e act_max:%.2e act_mean:%.2e act_std:%.2e'%(act_min, act_max, act_mean, act_std))
            act_map_norm = (act_map - act_thres_down) / (act_thres_up - act_thres_down) # normalize to [0, 1]
        elif map_method in ['individual']: # normalize firing rate of each neuron to [0, 1]
            act_map_norm = np.zeros(act_map.shape)
            nonzero_std = np.argwhere(act_std_N>0.0)
            zero_std = np.argwhere(act_std_N==0.0)
            act_map_norm[nonzero_std] = ( act_map[nonzero_std] - act_min_N[nonzero_std][:, np.newaxis, np.newaxis] ) / ((act_max_N - act_min_N)[nonzero_std][:, np.newaxis, np.newaxis])
            act_map_norm[zero_std] = self.get_relative_fire_rate(act_map[zero_std], act_thres_up, act_thres_down) # dead neurons and neurons with unchanging fire rate
        else:
            raise Exception('Agent.plot_act_map: invalid map_method: %s'%map_method)    
        
        cmap_func = plt.cm.get_cmap(cmap)
        act_map_mapped = cmap_func(act_map_norm) # [N_num, res_x, res_y, (r,g,b,a)]
        
        for i in range(act_map_mapped.shape[0]):
            ksize = 2 * int(res/25) + 1
            act_map_mapped[i,:,:,:] = cv.GaussianBlur(act_map_mapped[i,:,:,:], ksize=(ksize, ksize), sigmaX=1, sigmaY=1)

        res_x, res_y = act_map.shape[1], act_map.shape[2]
        #print('res_x:%d res_y:%d'%(res_x, res_y))
        arena_mask = arena.get_mask(res_x=res_x, res_y=res_y)
        arena_mask_white = (~arena_mask).astype(np.float)[:, :, np.newaxis] * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float)
        #print(arena_mask_white.shape)
        act_map_mapped = act_map_mapped * arena_mask.astype(np.float)[np.newaxis, :, :, np.newaxis] + arena_mask_white[np.newaxis, :, :, :]
        #print(act_map_mapped.shape)
        fig, ax = plt.subplots()
        ax.imshow(arena_mask.astype(np.float), cmap='jet')
        plt.savefig(save_path + 'mask_test.png')
        plt.close()

        if separate_ei:
            E_num, I_num, N_num = model.E_num, model.I_num, model.N_num
            print('E_num:%d I_num:%d N_num:%d'%(E_num, I_num, N_num))
            # plot E-neurons heat map
            if plot_num is None or plot_num in ['all']:
                plot_num_E = E_num
            else:
                plot_num_E = int(plot_num * E_num / N_num)
            #print('plot_num:%s plot_num_E:%s'%(str(plot_num), plot_num_E))
            if E_num < plot_num_E:
                plot_num_E = E_num
            if sample_method in ['random']:
                plot_index = np.sort(random.sample(range(E_num), plot_num_E))
            elif sample_method in ['grid', 'grid_score']:
                grid_score_E = grid_score[0:E_num]
                plot_index = np.argsort(-grid_score_E)[0:plot_num_E]
            save_info = {
                'save': save,
                'save_path': save_path,
                'save_name': '(E)' + save_name,
            }
            self.cat_rate_map_(arena, act_map_mapped, cmap, plot_num_E, plot_index, act_info, col_num, num_per_page, 'E-Neuron', grid_score, map_method, save_info)

            # plot I-Neurons heat map
            if plot_num is None or plot_num in ['all']:
                plot_num_I = I_num
            else:
                plot_num_I = int(plot_num * I_num / N_num)
            if plot_num_I > I_num:
                plot_num_I = I_num

            if sample_method in ['random']:
                plot_index = np.sort(random.sample(range(I_num), plot_num_I))
            elif sample_method in ['grid', 'grid_score']:
                grid_score_I = grid_score[E_num:]
                plot_index = np.argsort(-grid_score_I)[0:plot_num_I]
                plot_index += E_num
            else:
                raise Exception('Agent.plot_act_map: invalid sample method: %s'%sample_method)

            save_info = {
                'save': save,
                'save_path': save_path,
                'save_name': '(I)' + save_name,
            }            
            self.cat_rate_map_(arena, act_map_mapped, cmap, plot_num_I, plot_index, act_info, col_num, num_per_page, 'I-Neuron', grid_score, map_method, save_info)
        else:
            N_num = model.N_num
            if plot_num is None or plot_num in ['all']:
                plot_num = N_num
            if plot_num > N_num:
                plot_num = N_num
            if sample_method in ['random']:
                plot_index = np.sort(random.sample(range(self.N_num), plot_num))
            elif sample_method in ['grid', 'grid_score']:
                plot_index = np.argsort(-grid_score)[0:plot_num]
            save_info = {
                'save': save,
                'save_path': save_path,
                'save_name': save_name,
            }
            self.cat_rate_map_(arena, act_map_mapped, cmap, plot_num, plot_index, act_info, col_num, num_per_page, 'Neuron', grid_score, map_method, save_info)
    
    def cat_rate_map_(self, arena, act_map_mapped, cmap, plot_num, plot_index, act_info, col_num, num_per_page, \
        N_name='Neuron', grid_score=None, map_method=None, save_info=None):
        act_max, act_min, act_std = act_info['act_min'], act_info['act_max'], act_info['act_std']
        act_max_N, act_min_N, act_mean_N, act_std_N = act_info['act_max_N'], act_info['act_min_N'], act_info['act_mean_N'], act_info['act_std_N']
        act_min_N_ratio = (act_min_N - act_min) / (act_max - act_min)
        act_max_N_ratio = (act_max_N - act_min) / (act_max - act_min)
        act_range_N_ratio = (act_max_N - act_min_N) / (act_max - act_min)
        act_mean_N_ratio = (act_mean_N - act_min) / (act_max - act_min)
        
        save, save_path, save_name = save_info['save'], save_info['save_path'], save_info['save_name']

        page_num = plot_num // num_per_page
        if plot_num % num_per_page > 0:
            page_num += 1

        if map_method in ['universal', 'whole']:
            position_base = 1 # num of other plots such as colorbar before plotting act maps.
        else:
            position_base = 0
        
        print('plot_num:%d page_num:%d'%(plot_num, page_num))
        index_base = 0

        for page_index in range(page_num): # plot every page
            page_plot_num = min(num_per_page, plot_num - index_base)
            page_img_num = page_plot_num + position_base
            row_num = page_img_num // col_num
            if page_img_num % col_num > 0:
                row_num += 1
            #print('row_num:%d col_num:%d'%(row_num, col_num))
            plt.close()
            fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(5*col_num, 5*row_num)) # figsize unit: inches
            for i in range(page_plot_num):
                row_index, col_index = (i + position_base) // col_num, (i + position_base) % col_num
                N_index = plot_index[i + index_base]
                #print('page_plot_num:%d page_img_num:%d row_index:%d col_inedx:%d i:%d row_num:%d col_num:%d'
                #   %(page_plot_num, page_img_num, row_index, col_index, i, row_num, col_num))
                ax = get_ax(axes, row_index, col_index, row_num, col_num)
                
                im = ax.imshow(act_map_mapped[N_index, :, :], extent=(arena.x0, arena.x1, arena.y0, arena.y1))
                if grid_score is None:
                    ax.set_title('%s No.%d'%(N_name, N_index))
                else:
                    ax.set_title('%s No.%d score=%.2f (%.2f, %.2f)'%(N_name, N_index, grid_score[N_index], act_min_N[N_index], act_max_N[N_index]))
                ax_ = ax.inset_axes([1.05, 0.0, 0.12, 1.0])
                ax_.axis('off')
                ax_.add_patch(patches.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, linewidth=3.0, edgecolor='blue'))
                ax_.add_patch(patches.Rectangle((0.0, 1.0 - act_max_N_ratio[N_index]), 1.0, act_range_N_ratio[N_index], facecolor='blue')) # origin is on top-left corner
                ax_.add_line(Line2D([0.0, 1.0], [1.0 - act_mean_N_ratio[N_index], 1.0 - act_mean_N_ratio[N_index]], color='red'))
                ax.set_xticks(np.linspace(arena.x0, arena.x1, 5))
                ax.set_yticks(np.linspace(arena.y0, arena.y1, 5))
            
            for i in range(page_plot_num + position_base, row_num * col_num):
                row_index, col_index = (i + position_base) // col_num, (i + position_base) % col_num
                ax = get_ax(axes, row_index, col_index, row_num, col_num)
                ax.axis('off')

            if map_method in ['universal', 'whole']:
                ax = axes[0, 0]
                norm = mpl.colors.Normalize(vmin=act_info['act_thres_up'], vmax=act_info['act_thres_down'])
                #ax_ = fig.add_axes([0.0, 0.4, 1.0, 0.2]) # left, bottom, width, height. all are ratios to absolute coord range of canvas.
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
                #tick_locator = mpl.ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
                #cbar.locator = tick_locator
                #cbar.set_ticks(np.linspace(act_min, act_max, num=5))
                #cbar.update_ticks()
                cbar.set_label('Average fire rate', loc='center')

                ax.axis('off')
                '''
                fig.colorbar(im, cax=ax)
                ax.set_xlim(-0.10, 0.10)
                '''

            plt.tight_layout()
            if save:
                ensure_path(save_path)
                #cv.imwrite(save_path + save_name, imgs) # so that origin is in left-bottom corner.
                save_name = remove_suffix(save_name, '.png')
                plt.savefig(save_path + save_name + '(%d~%d)'%(index_base, min(index_base + num_per_page, plot_num)) + '.png')
            plt.close()
            
            index_base += num_per_page
    def plot_sample_num(self, point_count, arena, save=True, save_path='./', save_name='sample_num.png', cmap='jet'):
        fig, ax = plt.subplots()
        min_count, max_count = np.min(point_count), np.max(point_count)
        
        point_count_norm = (point_count - min_count) / (max_count - min_count)

        cmap_func = plt.cm.get_cmap(cmap)
        point_count_mapped = cmap_func(point_count_norm) # [N_num, res_x, res_y, (r,g,b,a)]

        norm = mpl.colors.Normalize(vmin=min_count, vmax=max_count)
        #ax_ = fig.add_axes([0.0, 0.4, 1.0, 0.2]) # left, bottom, width, height. all are ratios to absolute coord range of canvas.
        ax.imshow(point_count_mapped, extent=(arena.x0, arena.x1, arena.y0, arena.y1))
        #ax_ = ax.inset_axes([0.0, 0.4, 1.0, 0.2]) # left, bottom, width, height. all are ratios to sub-canvas of ax.
        ax.set_xticks(np.linspace(arena.x0, arena.x1, 5))
        ax.set_yticks(np.linspace(arena.y0, arena.y1, 5))
        cbar_ticks = np.linspace(min_count, max_count, num=5).astype(np.int)
        cbar_tick_labels = [str(tick) for tick in cbar_ticks.tolist()]
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        #tick_locator = mpl.ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
        #cbar.locator = tick_locator
        print(cbar_ticks)
        print(cbar_tick_labels)
        cbar.set_ticks(cbar_ticks)
        #cbar.update_ticks()
        cbar.set_ticklabels(cbar_tick_labels) # cbar.ax.set_yticklabels will cause Exception
        cbar.set_label('Total sample num', loc='center')
        ax.set_title('Sample Num When Plotting Rate Maps')

        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
    
    def plot_place_cells_prediction(self, act_map_info=None, model=None, arena=None, trainer=None, 
        res=50, save=True, save_path='./', save_name='place_cells_prediction.png'):

        model = self.model if model is None else model
        if model is None:
            raise Exception('Agent.plot_act_map: Error: model is None.')
        if arena is None:
            arena = self.arenas.get_current_arena()

        if act_map_info is None:
            batch_size = trainer.batch_size
            batch_num = int(trainer.batch_num / 100)
            act_map_info = self.cal_act_map(model=model, trainer=trainer, arena=arena, res=res, batch_size=batch_size, batch_num=batch_num) # 

        act_map, point_count = act_map_info['act_map'], act_map_info['point_count'] # act_map: [N_num, res_x, res_y]
        res_x, res_y = act_map.shape[1], act_map.shape[2]

        act_map = act_map.reshape([act_map.shape[0], res_x * res_y]) # [N_num, res_x * res_y]
        act_map = act_map.transpose((1, 0)) # [res_x * res_y, N_num]
        pc_map = model.get_output_from_act(act_map) # [res_x * res_y, place_cells_num]
        pc_map = pc_map.transpose((1, 0)) # [place_cells_num, res_x * res_y]
        pc_map = pc_map.reshape([self.place_cells.N_num, res_x, res_y])
        self.place_cells.plot_place_cells(act_map=pc_map, arena=arena, save_path=save_path, save_name='place_cells_predicted.png')
        '''
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
        '''
    def bind_arenas(self, arenas, index=None):
        self.arenas = arenas
        if hasattr(self, 'place_cells'):
            self.place_cells.bind_arenas(arenas, index=index)
    '''
    def bind_optimizer(self, optimizer):
        self.optimizer = optimizer
    '''
    def bind_model(self, model):
        self.model = model
        # to be modified
        if self.task in ['pc']:
            #self.cal_perform_ = self.cal_perform_pc
            pass
        elif self.task in ['coord']:
            #self.cal_perform_ = self.cal_perform_coord
            pass
        else:
            raise Exception('Invalid task: %s'%self.task)
    def anal(self, dict_, save_path='./', title='', items=['']):
        print('Agent: Analying...%s'%(title))
        print('Agent: Plotting agent path.')
        self.plot_path(save_path=save_path, save_name='path_plot.png', model=self.model, plot_num=2)
        print('Agent: Plotting act map.')
        self.anal_act(save_path=save_path,
                            model=self.model,
                            batch_size=dict_['batch_size'], batch_num=dict_['batch_num'],
                            arena=self.arenas.current_arena(),
                            separate_ei=self.model.separate_ei
                        )
    def save(self, save_path, save_name):
        ensure_path(save_path)
        with open(save_path + save_name, 'wb') as f:
            torch.save(self.dict, f)
        self.model.save(save_path, '%s_model'%save_name)