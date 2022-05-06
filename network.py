import random
import math
import json
import sys

# to allow training with just fetching last version of repository and restarting
sys.path.insert(0, '/home/ratingers/cf-tasks-rating/neural-networks')
from main import InitialWeightsGenerator
from activators import TanhActivator as Activator
sys.path.pop(0)

class IWGSubtime:
  def generate(self, iterable):
    for it in iterable: random.random()
    
    with open('/home/ratingers/cf-tasks-rating/neural-networks-subtime/weights.json', 'r') as f:
      return json.load(f)

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
    random.seed(0x14609A25)
    self.net2 = NeuralNetwork(Activator(), IWGSubtime(), 5, [4, 1])

  def predict(self, time, memory, lang):
    lang_const = 1

    if 'python' in lang:
      lang_const = 0
    elif 'pypy' in lang or 'java' in lang:
      lang_const = 0.5

    self.net.set_inputs((lang_const, 1 / (time + 1), math.log(memory + 1) / 21, 1))
    return 800 / next(self.net.calculate())

  def predict_v1(self, time, memory, lang, subtime):
    lang_const = 1

    if 'python' in lang:
      lang_const = 0
    elif 'pypy' in lang or 'java' in lang:
      lang_const = 0.5

    self.net2.set_inputs((lang_const, 1 / (time + 1), math.log(memory + 1) / 21, subtime / 18000, 1))
    return 800 / next(self.net2.calculate())
