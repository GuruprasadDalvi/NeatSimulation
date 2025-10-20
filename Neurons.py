        
from GeneticModels import Environment
import numpy as np
import math

class SensoryNeuron:
    def __init__(self,env: Environment) -> None:
        self.env = env
    def isInput(self):
        return 1
    def sense(self,**params):
        pass
    
    def isOutput(self):
        return 0
    
    
class InternalNeuron:
    def __init__(self) -> None:
        self.value = 0
        
    def add(self, weight, input):
        self.value += (weight*input)
        
    def activate(self):
        return float(np.tanh(self.value))
    def isInput(self):
        return 0
    
    def isOutput(self):
        return 0
    
class ResponseNeuron:
    def __init__(self,env: Environment) -> None:
        self.env = env
    def act(self, input_val):
        pass
    def isInput(self):
        return 0
    
    def isOutput(self):
        return 1
    
     
'''
    Sensory Neurons
'''        
class Skin(SensoryNeuron):
    def sense(self):
        return self.env.temp

class EyeNorth(SensoryNeuron):
    def sense(self, agent):
        return self.env.getNorthCellId(agent.x,agent.y)
    
class EyeSouth(SensoryNeuron):
    def sense(self, agent):
        return self.env.getSouthCellId(agent.x,agent.y)
    
class EyeEast(SensoryNeuron):
    def sense(self, agent):
        return self.env.getEastCellId(agent.x,agent.y)
    
class EyeWest(SensoryNeuron):
    def sense(self, agent):
        return self.env.getWestCellId(agent.x,agent.y)

class Stomach(SensoryNeuron):
    def sense(self, agent):
        return agent.energy/1000

    
'''
Response neurons
'''
class JointsX(ResponseNeuron):
    def act(self, input_val, agent):
        agent.x += round(input_val)

class JointsY(ResponseNeuron):
    def act(self, input_val, agent):
        agent.y += round(input_val)

class Mouth(ResponseNeuron):
    def act(self, input_val, env:Environment, agent):
        if input_val > 0 & env.isFood(agent.x,agent.y):
            agent.eat(agent.x,agent.y)
