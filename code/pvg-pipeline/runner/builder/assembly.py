from typing import Dict, List, Union, Callable, Tuple, Optional
from ignite.metrics import Metric
from ignite.engine import Events
from torchvision.transforms import Compose
import torch
from torch import Tensor

class Assembler():

    '''
    Base class for generic pipeline cration

    @param: references - list of uninitialized object (callable)
    @param: kwargs - list of keyword arguments as dicts
    
    @method: add - add a pair of callable references to objects and asscodated dict of kwargs
    @return: None

    @method: adds - adds a list of pairs of callable references to objects and asscodated dict of kwargs
    @return: None
    '''

    def __init__(self):

        # params
        self.references = []
        self.kwargs = []
        
    def add(self, reference: Callable, kwargs: Optional[Dict]=None)->None:

        #safety 
        assert isinstance(reference, Callable), "Input must be Callable"
        assert isinstance(kwargs, Dict) or kwargs is None, "Input must be Dict or NoneType"

        self.references.append(reference)
        self.kwargs.append(kwargs)

    def adds(self, references: List[Callable], kwargs: List[Optional[Dict]])->None:

        # safety
        assert isinstance(references, List), "Inputs must be of type List"
        assert isinstance(kwargs, List), "Inputs must be of type List"
        assert len(references) == len(kwargs), "Inputs must be of same lengths"

        for ref, kws in zip(references, kwargs):
            self.add(ref, kws)

class GenericAssembler(Assembler):

    '''
    Generic extension of Assembler	
    @method: build - initializes references with kwargs
    @return: List of initialized objects 
    '''

    def __init__(self):
        super().__init__()

    def build(self)->List:
        objects = []
        for ref, kws in zip(self.references, self.kwargs):
            if kws is None: objects.append(ref())
            if kws is not None: objects.append(ref(**kws))
        
        if len(objects) == 0: return None
        else: return objects

class TransformAssembler(Assembler):
	
    '''
    Generic extension of Assembler for a transformation pipeline
    
    @method: build - initializes references with kwargs
    @return: Callable transform pipeline
    @return params: Tensor returns Tensor

    Note: please consider wrappers for data synergy
    '''

    def __init__(self):
            super().__init__()

    def build(self)->Callable:
        transforms = []
        for ref, kws in zip(self.references, self.kwargs):
            if kws is None: transforms.append(ref())
            if kws is not None: transforms.append(ref(**kws))

        if len(transforms) == 0: return None
        
        transform_pipeline = Compose(transforms)
        
        @torch.no_grad()
        def transformer(inputs: Union[Dict, Tensor])->Union[Dict, Tensor]:
            return transform_pipeline(inputs)

        return transformer

class InputAssembler(Assembler):

    '''
    Generic extension of Assembler with additional arg
    
    @method: build - initializes references with input and kwargs
    @return: List of initialized objects 
    '''

    def __init__(self):
        super().__init__()

    def build(self, args)->List:
        objects = []
        for ref, arg, kws in zip(self.references, args, self.kwargs):
            if kws is None: objects.append(ref(arg))
            if kws is not None: objects.append(ref(arg, **kws))

        if len(objects) == 0: return None
        else: return objects

class MetricAssembler():
	
    '''
    simple assembler based on assembly with custom attributes
    naming convension fix
    '''

    def __init__(self):

        # params
        self.metrics = []
        self.names = []
        
    def add(self, metric: Metric, name: str)->None:

        #safety 
        assert isinstance(metric, Metric), "Metric must be of type Metric"
        assert isinstance(name, str), "Name must be of type str"

        self.metrics.append(metric)
        self.names.append(name)

    def adds(self, metrics: List[Callable], names: List[str])->None:

        # safety
        assert isinstance(metrics, List), "Must add List of Metrics"
        assert isinstance(names, List), "Must add List of strs"
        assert len(metrics) == len(names), "Inputs must be of same length"

        for metric, name in zip(metrics, names):
            self.add(metric, name)

    def build(self)->Tuple:
        if len(self.metrics) == 0 or len(self.names) == 0: return None
        return (self.metrics, self.names)

class HandlerAssembler():
    '''
    simple assembler based on assembly with custom attributes
    naming convension fix
    '''

    def __init__(self):

        # params
        self.handlers = []
        self.states = []

    def add(self, handler: Callable, state: Events)->None:

        #safety 
        assert isinstance(handler, Callable), "Handler must be of type Handler"
        assert isinstance(state, Events), "State must be of type Events"

        self.handlers.append(handler)
        self.states.append(state)

    def adds(self, handlers: List[Callable], states: List[Events])->None:

        # safety
        assert isinstance(handlers, List), "Must add List of Callable handlers"
        assert isinstance(states, List), "Must add List of Events states"
        assert len(handlers) == len(states), "Inputs must be of same length"

        for handler, state in zip(handlers, states):
            self.add(handler, state)

    def build(self)->Tuple:
        if len(self.handlers) == 0 or len(self.states) == 0: return None
        return (self.handlers, self.states)
