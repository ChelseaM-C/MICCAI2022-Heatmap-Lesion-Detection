# Type checks
import torch
from torch.nn import Module
from typing import Dict
from ignite.engine import Engine

# System imports
import sys
import os

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

from constants.pipeline import CheckpointKeys as CP_KEYS

BEST_STAT_STR = 'best_stat'
MIN_STR = 'min'
MAX_STR = 'max'

class Checkpoint():
    def __init__(self, 
                 rootdir: str, 
                 name: str, 
                 objects: Dict,
                 patience: int, 
                 period: int,
                 metric_name: str,
                 score_fn: str='max',
                 save_period: int=0):
        
        '''
            Checkpoint experiment
                - checkpoint best model given a metric type
                - perform scheduled checkpoints after an initial patience period
                - perform scheduled checkpoints on a periodic basis
                
            Saves checkpoint to disk
            
            Assumes struct of checkpointed obj in form:
            {'obj_name': obj} where all obj are serializable via obj.state_dict
            
            Multimodel functionality
            {'model_a' : model_a, 'model_b' : model_b, ....}
            
            All objects must be contained in a dict, regardless of size
        '''
        
        self.rootdir = rootdir
        self.name = name
        self.metric = metric_name
        self.objects = objects
        self.patience = patience 
        self.period = period 
        self.score_fn = score_fn
        self.stat = None
        self.save_period = save_period
        
        if self.score_fn not in [MIN_STR, MAX_STR]:
            raise ValueError("Score function should be 'min' or 'max'")
            
    def __call__(self, 
                 engine: Engine)->None:
                
        # warm-up period
        if engine.state.epoch < self.patience:
            return 
        
        if self.stat is None:
            self.stat = engine.state.metrics[self.metric]
        
        # best configuration considerations
        isSelected = False
        if self.score_fn == MAX_STR:
            if engine.state.metrics[self.metric] >= self.stat:
                self.stat = engine.state.metrics[self.metric]
                isSelected = True
                
        elif self.score_fn == MIN_STR:
            if engine.state.metrics[self.metric] <= self.stat:
                self.stat = engine.state.metrics[self.metric]
                isSelected = True
        
        # log configuration
        save_on_period = False
        if self.save_period > 0:
            if engine.state.epoch % self.save_period == 0:
                save_on_period = True

        if isSelected or engine.state.epoch % self.period == 0 or save_on_period:
            state_checkpoint = {}
            model_checkpoint = {}

            for object_name, object_dict in self.objects.items():
                if object_name == CP_KEYS.MODELS:
                    model_checkpoint[object_name] = { k: v.state_dict() for k, v in object_dict.items() }
                elif object_name == CP_KEYS.ENGINE:
                    state_checkpoint[object_name] = object_dict.state_dict()
                else:
                    state_checkpoint[object_name] = { k: v.state_dict() for k, v in object_dict.items() }
            state_checkpoint[BEST_STAT_STR] = self.stat

            # checkpoint config
            if engine.state.epoch % self.period == 0:
                torch.save(state_checkpoint, os.path.join(self.rootdir, self.name + '_state_checkpoint.pth.tar'))
                torch.save(model_checkpoint, os.path.join(self.rootdir, self.name + '_model_checkpoint.pth.tar'))
            
            # log best config
            if isSelected:
                torch.save(state_checkpoint, os.path.join(self.rootdir, self.name + '_state_best.pth.tar'))
                torch.save(model_checkpoint, os.path.join(self.rootdir, self.name + '_model_best.pth.tar'))

            if save_on_period:
                epoch = str(engine.state.epoch)
                torch.save(state_checkpoint, os.path.join(self.rootdir, self.name + '_state_'+epoch+'.pth.tar'))
                torch.save(model_checkpoint, os.path.join(self.rootdir, self.name + '_model_'+epoch+'.pth.tar'))
                
    def resume(self, 
               state_path: str, 
               model_path: str,
               objects=None)->None:

        if objects is None: objects = self.objects
        state_checkpoint = torch.load(state_path, map_location='cpu')
        model_checkpoint = torch.load(model_path, map_location='cpu')

        for state_name, state_value in state_checkpoint.items():
            if state_name == BEST_STAT_STR:
                self.stat = state_value
            elif state_name == CP_KEYS.ENGINE:
                objects[state_name].load_state_dict(state_value)
            else:
                for k, v in state_value.items():
                    objects[state_name][k].load_state_dict(v)

        for k, v in model_checkpoint[CP_KEYS.MODELS].items():
            objects[CP_KEYS.MODELS][k].load_state_dict(v)
        
    @staticmethod
    def load_model(path: str, 
                   model: Module,
                   name_key: str)->None:

        model_checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(model_checkpoint[CP_KEYS.MODELS][name_key])
    
    @staticmethod
    def load_models(path: str,
                    models: dict)->None:
        
        if not isinstance(models, dict):
            raise ValueError("Input 'models' must be a dict")

        model_checkpoint = torch.load(path, map_location='cpu')
        for k, v in model_checkpoint[CP_KEYS.MODELS].items():
            models[k].load_state_dict(v)
