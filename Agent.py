from nNet import nNetwork
from random import randint,uniform
from pygame import draw
from pygame import gfxdraw
import math
from copy import deepcopy
import threading

MAX_TEMP = 50
MIN_TEMP = 1

def dist(p1,p2):
    return math.dist(p1,p2)


        
class Food:
    def __init__(self,screen) -> None:
        self.x = randint(0,800)
        self.y = randint(0,800)
        self.energy = 10
        self.color = (150,250,150)
        self.screen = screen
        
    def render(self):
        # gfxdraw.aacircle(self.screen,[0,0,0],(self.x,self.y),self.energy+1)
        gfxdraw.aacircle(self.screen,self.x,self.y,self.energy+1,[0,0,0])
        # gfxdraw.aacircle(self.screen,self.color,(self.x,self.y),self.energy)
        gfxdraw.filled_circle(self.screen,self.x,self.y,self.energy,self.color)
        gfxdraw.aacircle(self.screen,self.x,self.y,self.energy,self.color)


class Agent:
    def __init__(self,screen,env) -> None:
        self.brain = nNetwork([10,10,6],activeFun="tanh")
        self.x = randint(0,800)
        self.y = randint(0,800)
        self.color = [randint(50,255),50,randint(50,255)]
        self.energy = 1000
        self.screen = screen
        self.days = 0
        self.max_vision = 50
        self.radius = 5
        self.env = env
        
    def update(self):
        
        # self.move(round(float(op[0])),round(float(op[1])),round(float(op[2]*10)))
        self.energy-=1
        if self.energy>2000:
            self.reproduce()
    
    def move(self, dx, dy, acc):
        self.x += dx*acc
        self.y += dy*acc
        self.x = self.x%800
        self.y = self.y%800
        if(self.x<0):
            self.x = 800
        if(self.y<0):
            self.y = 800
        self.energy-=1
    
    def reproduce(self):
        pass
    
    def attack(self):
        pass
        
        
        
    def mutate(self):
        # self.color[-1]=(self.color[-1]+randint(-1,1))%256
        self.brain.mutate()
        
        
    def render(self):
        if self.energy<0:
            self.energy = 0
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l1[0]),int(self.l1[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l2[0]),int(self.l2[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l3[0]),int(self.l3[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l4[0]),int(self.l4[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l5[0]),int(self.l5[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l6[0]),int(self.l6[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l7[0]),int(self.l7[1]),self.color)
        # gfxdraw.line(self.screen,int(self.x),int(self.y),int(self.l8[0]),int(self.l8[1]),self.color)
        gfxdraw.filled_circle(self.screen,int(self.x),int(self.y),int(self.radius),self.color)
        gfxdraw.aacircle(self.screen,int(self.x),int(self.y),int(self.radius),self.color)