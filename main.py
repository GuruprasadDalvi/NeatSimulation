import time
from copy import deepcopy
import pygame
import random

from GeneticModels import Environment, Agent, Food

# Initialize Pygame
pygame.init()
WIDTH,HEIGH = 1000,800
CELL_SIZE = 16
screen = pygame.display.set_mode((WIDTH,HEIGH))





# Create window
pygame.display.set_caption("Simulation")
running = True

# Window size
env = Environment(WIDTH,HEIGH,CELL_SIZE,screen)
for i in range(env.grid_size//CELL_SIZE-1):
    for j in range(env.grid_size//CELL_SIZE-1):
            a = Agent(env, screen)
            env.add_agent(a)

for i in range(len(env.agents)):
    f = Food(env, screen)
    env.add_food(f)
        
            
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x,y = pygame.mouse.get_pos()
            env.select(x,y)
        
    env.update()
    
    # Draw environment
    screen.fill(env.bg_color)
    env.render()
    pygame.display.flip()
    # time.sleep(1)
    # print("=========================")
    pygame.display.set_caption(f"Simulation Population: {len(env.agents)} Food: {len(env.foods)} Max Age: {env.maxAge} Env Age: {env.age}")
    if(len(env.agents)<10):
        for i in range(env.grid_size//CELL_SIZE -1):
            for j in range(env.grid_size//CELL_SIZE -1):
                if (i)%2==0 and (j)%2==0:
                    a = Agent(env, screen)
                    a.x=i
                    a.y=i
                    env.add_agent(a)
        

pygame.quit()
