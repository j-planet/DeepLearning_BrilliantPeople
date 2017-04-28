from tensorflow import name_scope
from abc import ABCMeta, abstractmethod

from utilities import str_2_activation_function


class AbstractLayer(metaclass=ABCMeta):

    def __init__(self, activation=None, loggerFactory=None):

        with name_scope(self.__class__.__name__):

            self.print = print if loggerFactory is None else loggerFactory.getLogger('Model').info
            self.print('Constructing: ' + self.__class__.__name__)
            self.activationFunc = str_2_activation_function(activation)

            self.make_graph()

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, val):
        self.__output = val

    @abstractmethod
    def make_graph(self):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    @abstractmethod
    def output_shape(self, inputDim_):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)
