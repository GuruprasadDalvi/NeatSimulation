import random
import base64
from pygame import draw, Rect
from pygame import gfxdraw
from nNet import nNetwork
from pygame.font import Font
import copy

def interpolate(percent, min_value, max_value):
    return min_value + percent * (max_value - min_value)

class Environment:
    def __init__(self,wid,hei,cell_size,screen) -> None:
        self.wid = wid
        self.hei = hei
        self.cell_size = cell_size
        self.screen = screen
        self.grid = []
        self.grid_size = min(wid,hei)
        for i in range((self.grid_size//cell_size)):
            row = []
            for j in range((self.grid_size//cell_size)):
                d = DefaultObj(self, j,i)
                row.append(d)
            self.grid.append(row)
        # self.sound_grid = [[0]*(wid//cell_size)]*(hei//cell_size)
        # self.food_grid = [[0]*(wid//cell_size)]*(hei//cell_size)
        self.border_color = [10,10,10]
        self.temp = 0
        self.bg_color  = [10,10,50]
        self.selected = None
        self.font = Font(None, 16)
        self.agents = []
        self.foods = []
        self.increasing = True
        self.maxAge = 0
        self.age = 0
        # Simulation control
        self.paused = False
        self._request_step = False
        
    def select(self,x,y):
        x=x//self.cell_size
        y=y//self.cell_size
        if type(self.grid[x][y])==Agent:
            self.selected = self.grid[x][y]
        
        # print(f"SelectedL {self.grid[x][y].objId}  {self.selected}")
        # print(f"North To selected: {self.getNorthCellId(x,y)}")
        # print(f"East To selected: {self.getEastCellId(x,y)}")
        # print(f"South To selected: {self.getSouthCellId(x,y)}")
        # print(f"West To selected: {self.getWestCellId(x,y)}")
        # print(f"NorthEast To selected: {self.getNorthEastCellId(x,y)}")
        # print(f"SouthEast To selected: {self.getSouthEastCellId(x,y)}")
        # print(f"SouthWest To selected: {self.getSouthWestCellId(x,y)}")
        # print(f"NorthWest To selected: {self.getNorthWestCellId(x,y)}")
        
    def render(self):
        # Update agents only if running or a single step is requested
        if not self.paused or self._request_step:
            for a in list(self.agents):
                a.update()
                if a.age>self.maxAge:
                    self.maxAge = a.age
                    self.selected = a
            self.age+=1
            # clear step request after processing one tick
            self._request_step = False
        for x in range(0, self.grid_size, self.cell_size):
            for y in range(0, self.grid_size, self.cell_size):
                a = self.grid[y//self.cell_size][x//self.cell_size]
                a.render()
                
                # Render the x, y coordinates on the cell
        if self.selected:
                    a = self.selected
                    if a and type(a)==Agent:
                        texts = [
                            f"Attack:    {a.attacked} ",
                            f"Age:       {round(a.age,4)}",
                            f"Children:  {round(a.child,4)}",
                            f"Energy:    {round(a.energy,4)}",
                            f"Traveled:  {round(a.traveled,4)}",
                            f"X:  {round(a.x)}",
                            f"Y:  {round(a.y)}",
                        ]
                        yPos = 20
                        for t in texts:
                            text = self.font.render(t, True, (255, 255, 255))
                            text_rect = text.get_rect()
                            text_rect.center = (int(self.wid*.9), yPos)
                            self.screen.blit(text, text_rect)
                            yPos+=25
        
    def isCellEmpty(self, x,y):
        try:
            return type(self.grid[x][y])==DefaultObj 
        except:
            True   
      
    def update(self):
        if self.increasing:
            self.temp += 0.25
            if self.temp >= 100:
                self.temp = 100
                self.increasing = False
        else:
            self.temp -= 0.25
            if self.temp <= 0:
                self.temp = 0
                self.increasing = True
        self.bg_color  = [20,20,50]
        if len(self.foods)<(len(self.agents)*.5) or len(self.foods)<1000:
            for i in range(int(1000-len(self.foods))):
                f = Food(self, self.screen)
                self.add_food(f)

    def toggle_pause(self):
        self.paused = not self.paused

    def step(self):
        # Request a single simulation tick while paused
        if self.paused:
            self._request_step = True
    
    def getNorthCellId(self,x,y):
        try:
            return self.grid[x][y-1].objId
        except:
            return -1.0
    
    def getSouthCellId(self,x,y):
        try:
            return self.grid[x][y+1].objId
        except:
            return -1.0
    
    def getEastCellId(self,x,y):
        try:
            return self.grid[x+1][y].objId
        except:
            return -1.0
    
    def getWestCellId(self,x,y):
        try:
            return self.grid[x-1][y].objId
        except:
            return -1.0
      
    def getNorthEastCellId(self,x,y):
        try:
            return self.grid[x+1][y-1].objId
        except:
            return -1.0
    
    def getSouthEastCellId(self,x,y):
        try:
            return self.grid[x+1][y+1].objId
        except:
            return -1.0
    
    def getSouthWestCellId(self,x,y):
        try:
            return self.grid[x-1][y+1].objId
        except:
            return -1.0
    
    def getNorthWestCellId(self,x,y):
        try:
            return self.grid[x-1][y-1].objId
        except:
            return -1.0
    
    def damage(self,x,y,amount):
        try:
            a = self.grid[x][y]
            if type(a)==Agent:
                a.energy -=amount
                if(a.energy<0):
                    self.grid[a.x][a.y] = DefaultObj(self,a.x, a.y)
                    self.agents.remove(a)
                    if self.selected == a:
                        self.selected = None 
        except:
            pass
    
    def ifFood(self, x,y):
        pass
    
    def add_agent(self, agent):
        if type(self.grid[agent.x][agent.y]) == DefaultObj:
            self.agents.append(agent)
            self.grid[agent.x][agent.y]= agent

    def add_food(self, food):
        if type(self.grid[food.x][food.y]) == DefaultObj:
            self.grid[food.x][food.y]= food
            self.foods.append(food)

class DefaultObj:
    def __init__(self, env: Environment,x,y) -> None:
        self.env = env
        self.objId = -1.0
        self.x = x
        self.y = y
    
    def render(self):
        rect = Rect(self.x*self.env.cell_size, self.y*self.env.cell_size, self.env.cell_size, self.env.cell_size)
        draw.rect(self.env.screen, self.env.border_color, rect, 1)
        
    def update(self):
        pass

class Agent:
    
    def __init__(self, env: Environment, screen) -> None:
        self.x = random.randint(0,len(env.grid[0])-1)
        self.y = random.randint(0,len(env.grid)-1)
        self.env = env
        self.objId = 0.5
        self.screen = screen
        self.color = [random.randint(0,155),random.randint(0,155),random.randint(0,155)]
        self.font = Font(None, 16)
        self.brain = nNetwork([12,10,6],activeFun="tanh")
        self.energy = 2000
        self.maxenergy = 5000
        self.attacked = 0
        self.child = 0
        self.age = 0
        self.traveled = 0
  
    def update(self):
        self.age+=1
        preds = self.brain.feedforward([
            self.env.getNorthCellId(self.x,self.y),
            self.env.getNorthEastCellId(self.x,self.y),
            self.env.getEastCellId(self.x,self.y),
            self.env.getSouthEastCellId(self.x,self.y),
            self.env.getSouthCellId(self.x,self.y),
            self.env.getSouthWestCellId(self.x,self.y),
            self.env.getWestCellId(self.x,self.y),
            self.env.getNorthWestCellId(self.x,self.y),
            self.energy/self.maxenergy,
            self.env.temp,
            self.x/len(self.env.grid[0]),
            self.y/len(self.env.grid[0]),
            ])
        
        # Attack
        if float(preds[2])>0:
            self.attack()
        # Reproduce 
        if(self.energy>2000) and preds[3]>0:
            self.reproduce()
            self.energy-=100
        
        # Eat
        if(float(preds[4]))>0:
            self.eat()
            if self.energy>self.maxenergy:
                self.energy = self.maxenergy
            
        
        if(self.energy<0):
            self.env.grid[self.x][self.y] = DefaultObj(self.env,self.x, self.y)
            self.env.agents.remove(self)
            if self.env.selected == self:
                self.env.selected = None 
        elif self.age>200 and self.traveled < 5:
            self.env.grid[self.x][self.y] = DefaultObj(self.env,self.x, self.y)
            self.env.agents.remove(self)
            if self.env.selected == self:
                self.env.selected = None 
        else:
            self.energy-=1*(self.env.temp/10)
            self.move(round(float(preds[0])),round(float(preds[1])))
        
        
        # pass

    def move(self,dx,dy):
        if self.env.isCellEmpty(self.x+dx,self.y+dy):
            df = DefaultObj(self.env, self.x,self.y)
            newX = (self.x + dx)%len(self.env.grid[self.x])
            newY = (self.y + dy)%len(self.env.grid[self.y])
            self.env.grid[newX][newY] = self
            self.env.grid[self.x][self.y] = df
            self.x= newX
            self.y= newY
            self.traveled+=1
        
    def reproduce(self):
        newX = random.randint(0,self.env.grid_size//self.env.cell_size -1)
        newY = random.randint(0,self.env.grid_size//self.env.cell_size -1)
        a = Agent(self.env,self.screen)
        a.x = newX
        a.y = newY
        a.brain = copy.deepcopy(self.brain)
        a.brain.mutate()
        a.color = self.color
        self.env.add_agent(a)
        self.child+=1
    
    def attack(self):
            self.energy-=80
            self.env.damage(self.x,self.y-1,20)
            self.env.damage(self.x,self.y+1,20)
            
            self.env.damage(self.x-1,self.y,10)
            self.env.damage(self.x+1,self.y,10)
            
            self.env.damage(self.x-1,self.y-1,5)
            self.env.damage(self.x+1,self.y+1,5)
            self.env.damage(self.x+1,self.y-1,5)
            self.env.damage(self.x-1,self.y+1,5)
            self.attacked+=1

    def render(self):
        rect = Rect(self.x*self.env.cell_size, self.y*self.env.cell_size, self.env.cell_size, self.env.cell_size)
        draw.rect(self.screen, self.color, rect)
        if self.env.selected==self:
            rect2 = Rect((self.x*self.env.cell_size) +int(self.env.cell_size*.25), (self.y*self.env.cell_size) +int(self.env.cell_size*.25) , int(self.env.cell_size*.5), int(self.env.cell_size*.5))
            draw.rect(self.screen, [255,0,0], rect2)
        # draw.circle(self.screen,self.color,[self.x*self.env.cell_size,self.y*self.env.cell_size],self.env.cell_size//2)
        
        # rect = Rect(self.x*self.env.cell_size-10, self.y*self.env.cell_size-10, int(self.env.cell_size*1.5), 5)
        # draw.rect(self.screen, [255,255,255], rect)
        
        # rect = Rect(self.x*self.env.cell_size-10, self.y*self.env.cell_size-10, int(int(self.env.cell_size*1.5)*(self.energy/self.maxenergy)), 5)
        # draw.rect(self.screen, [0,55,0], rect)
        
        # text = self.font.render(f"({self.x , self.y})", True, (255, 255, 255))
        # text_rect = text.get_rect()
        # text_rect.center = (self.y + self.env.cell_size//2, self.x + self.env.cell_size//2)
        # self.screen.blit(text, text_rect)
    
    def eat(self):
        # East
        try:
            if type(self.env.grid[self.x+1][self.y])==Food:
                self.energy+=self.env.grid[self.x+1][self.y].energy
                self.env.foods.remove(self.env.grid[self.x+1][self.y])
                self.env.grid[self.x+1][self.y] = DefaultObj(self.env,self.x+1, self.y)
                return
            # North
            if type(self.env.grid[self.x][self.y-1])==Food:
                self.energy+=self.env.grid[self.x][self.y-1].energy
                self.env.foods.remove(self.env.grid[self.x][self.y-1])
                self.env.grid[self.x][self.y-1] = DefaultObj(self.env,self.x, self.y-1)
                return
            
            # West
            if type(self.env.grid[self.x-1][self.y])==Food:
                self.energy+=self.env.grid[self.x-1][self.y].energy
                self.env.foods.remove(self.env.grid[self.x-1][self.y])
                self.env.grid[self.x-1][self.y] = DefaultObj(self.env,self.x-1, self.y)
                return
            
            # South
            if type(self.env.grid[self.x][self.y+1])==Food:
                self.energy+=self.env.grid[self.x][self.y+1].energy
                self.env.foods.remove(self.env.grid[self.x][self.y+1])
                self.env.grid[self.x][self.y+1] = DefaultObj(self.env,self.x, self.y+1)
                return
        except:
            pass
            
        
        
class Food:
    
    def __init__(self, env, screen) -> None:
        self.x = random.randint(0,len(env.grid[0])-1)
        self.y = random.randint(0,len(env.grid)-1)
        self.energy = random.randint(10,100)
        self.env = env
        self.objId = 1.0
        self.screen = screen
    
    
    def render(self):
        rect = Rect(self.x*self.env.cell_size, self.y*self.env.cell_size, self.env.cell_size, self.env.cell_size)
        draw.rect(self.screen, [0,0,255], rect)
        
    def update(self):
        pass