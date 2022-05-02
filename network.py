import random
import math

import sys

# to allow training with just fetching last version of repository and restarting
if '/home/ratingers/cf-tasks-rating/neural-networks' not in sys.path:
  sys.path.insert(0, '/home/ratingers/cf-tasks-rating/neural-networks')

from main import InitialWeightsGenerator
from activators import TanhActivator as Activator

# class InitialWeightsGenerator:
#   INIT_WEIGHTS = [0.715,  4.634,  0.812,  0.787,
# -0.539,-0.340,-0.541,-0.031,  -0.215,3.112,-0.301,0.330,  0.037,0.049,1.415,0.701,
# 0.284,2.694,0.967].__iter__()
#
#   def generate(self, iterable):
#     if not self.INIT_WEIGHTS:
#       return [random.random() * 2 - 1 for it in iterable]
#     else:
#       return [random.random() * 0 + self.INIT_WEIGHTS.__next__() for it in iterable]

class INeuron:
  def __init__(self, activator, initializer, previous_layer): pass
  def calculate(self):       pass
  def sprintf_weights(self): pass

#####################

class InputValue(INeuron):
  def __init__(self, activator, initializer, previous_layer):
    self.activator = activator
    self.coef = initializer.generate(' ')[0]
    self.value = 0
    self.cache = None

  def calculate(self):
    if not self.cache:
      self.cache = self.activator.result(self.value)
    return self.cache

#####################

class Neuron(INeuron):
  def __init__(self, activator, initializer, previous_layer):
    self.activator = activator
    self.previous_layer = previous_layer
    self.next_layer = None
    self.coefs = initializer.generate(previous_layer)
    self.cache = None

  def calculate(self):
    if not self.cache:
      s = sum(self.coefs[i] * v.calculate() for i, v in enumerate(self.previous_layer))
      self.cache = self.activator.result(s)

    return self.cache

#####################

class NeuralNetwork:
  def __init__(self, activator, initializer, inputs, layer_sizes):
    self.layers = [[InputValue(activator, initializer, None) for i in range(inputs)]]

    self.activator_prefix = str(activator)

    for size in layer_sizes:
      self.layers.append([Neuron(activator, initializer, self.layers[-1]) for i in range(size)])
      for neuron in self.layers[-2]:
        neuron.next_layer = self.layers[-1]

  def set_inputs(self, input_values):
    for i, neuron in enumerate(self.layers[0]):
      neuron.value = input_values[i]

    for layer in self.layers:
      for neuron in layer:
        neuron.cache = None

  def calculate(self):
    for neuron in self.layers[-1]:
      yield neuron.calculate()

#####################

class Predictor:
  def __init__(self):
    random.seed(0x14609A25)
    self.net = NeuralNetwork(Activator(), InitialWeightsGenerator(), 4, [3, 1])

  def predict(self, time, memory, lang):
    lang_const = 1

    if 'python' in lang:
      lang_const = 0
    elif 'pypy' in lang or 'java' in lang:
      lang_const = 0.5

    self.net.set_inputs((lang_const, 1 / (time + 1), math.log(memory + 1) / 21, 1))
    return 800 / next(self.net.calculate())
