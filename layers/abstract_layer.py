from tensorflow import name_scope
from abc import ABCMeta, abstractmethod

from utilities import str_2_activation_function


class AbstractLayer(metaclass=ABCMeta):

    def __init__(self, input_, inputDim_, activation=None, loggerFactory=None):
        """
        :type inputDim_: tuple
        """

        with name_scope(self.__class__.__name__):

            self.loggerFactory = loggerFactory
            self.print = print if loggerFactory is None else loggerFactory.getLogger(self.__class__.__name__).info
            self.inputDim = inputDim_
            self.input = input_     # does not work if called after self.inputDim is set

            self.print('Constructing: ' + self.__class__.__name__)
            self.activationFunc = str_2_activation_function(activation)

            self.make_graph()

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, val):
        self.__output = self.activationFunc(val)

    @abstractmethod
    def make_graph(self):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError('This (%s) is an abstract class.' % self.__class__.__name__)

    def input_modifier(self, val):
        return val

    @property
    def input(self):
        return self.__input

    @input.setter
    def input(self, val):
        self.__input = self.input_modifier(val)

    @classmethod
    @abstractmethod
    def new(cls, **kwargs):
        raise NotImplementedError('This (%s) is an abstract class.' % cls.__name__)
