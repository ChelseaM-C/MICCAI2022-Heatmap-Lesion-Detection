from typing import List, Union, Optional, Callable, Tuple
from ignite.engine.engine import Engine
from ignite.metrics import Metric
from ignite.engine import Events
import warnings

class EngineUtility:

    '''
    Utility class for assembling an engine and associated workflow
    Supports:
            building engine
            assembling engine
            attaching metrics
            attaching handlers
            syncing engines
    '''

    '''
    @method: attach_metrics - attach a metric or list of metrics to an engine
    @param: engine
    @param: metics List or Object
    @param: names List or String
    @return: None
    '''
    @staticmethod
    def attach_metrics(engine: Engine, 
                       metrics: Union[List, Metric],
                       names: Union[List, str])->None:

        if isinstance(metrics, List) and isinstance(names, List):
            assert(len(metrics)==len(names))
            for metric, name in zip(metrics, names):
                metric.attach(engine, name)
        else:
            metrics.attach(engine, names)

    '''
    @method: attach_handlers - - attach a handler or list of handlers to an engine
    @param: engine
    @param: handlers List or Object
    @param: states List or String
    @return: None
    '''
    @staticmethod
    def attach_handlers(engine: Engine, 
                        handlers: Union[List, Metric],
                        states: Union[List, Events])->None:
            
        if isinstance(handlers, List) and isinstance(states, List):
            assert(len(handlers)==len(states))
            for handler, state in zip(handlers, states):
                engine.add_event_handler(state, handler)
        else:
            engine.add_event_handler(states, handlers)


    '''
    @method: build - build a generic workflow
    @param: workflow
    @return: engine
    '''
    @staticmethod
    def build(process: Callable)->Engine:
        engine = Engine(process)
        return engine

    '''
    @method: assemble - build a generic workflow
    @param: process
    @param: metics List or Object
    @param: metric_name List or String
    @param: handlers List or Object
    @param: states List or String
    @return: engine
    '''
    @staticmethod
    def assemble(process: Callable,
                 metrics: Optional[List[Metric]]=None, 
                 metric_names: Optional[List[str]]=None,
                 handlers:Optional[Tuple[Callable, str]] = None,
                 states: Optional[List[Events]]=None)->Engine:
        
        engine = Engine(process)
        if metric_names is not None and metrics is not None: EngineUtility.attach_metrics(engine, metrics, metric_names)
        else: warnings.warn('Warning Message: no metrics attached')
        if handlers is not None and states is not None: EngineUtility.attach_handlers(engine, handlers, states)
        else: warnings.warn('Warning Message: no handlers attached')

        return engine

    '''
    Sync state of engines
    Supports:
            -epoch
    '''
    @staticmethod
    def sync_engine(source: Engine,
                    target: Engine)->None:
        target.state.epoch = source.state.epoch
