import time
from copy import deepcopy
import pygame
import random
import csv
import os

from GeneticModels import Environment, Agent, Food, Predator

# Initialize Pygame
pygame.init()
# Widen the window to give the right-side panels more room without changing grid size
WIDTH,HEIGH = 1400,800
CELL_SIZE = 16
screen = pygame.display.set_mode((WIDTH,HEIGH))




log_path = "population_log.csv"
with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age", "population"])
            
            
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
    
# Seed some predators
for i in range(100):
    p = Predator(env, screen)
    env.add_predator(p)
        
            
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
    # Elapsed time since start, formatted to seconds like age indicator
    elapsed = getattr(env, 'elapsed_seconds', 0.0)
    # Average age of agents
    avg_age = int(sum(a.age for a in env.agents) / len(env.agents)) if env.agents else 0
    pygame.display.set_caption(
        f"Simulation Population: {len(env.agents)} "
        f"Predators: {len(getattr(env, 'predators', []))} "
        f"Food: {len(env.foods)} "
        f"Max Age: {env.maxAge} "
        f"Avg Age: {avg_age} "
        f"Env Age: {env.age} "
        f"Elapsed: {int(elapsed)}s"
    )
    # Log population to CSV every 25 steps
    if env.age > 0 and env.age % 10 == 0:
        file_exists = os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["age", "population"]) 
            writer.writerow([env.age, len(env.agents)])
    if(len(env.agents)<50):
        for i in range(env.grid_size//CELL_SIZE -1):
            for j in range(env.grid_size//CELL_SIZE -1):
                if len(env.agents)<1000 :
                    if random.random()<0.5:
                
                        a = Agent(env, screen)
                        a.x=random.randint(0,env.grid_size//CELL_SIZE -1)
                        a.y=random.randint(0,env.grid_size//CELL_SIZE -1)
                        env.add_agent(a)
                    else:
                        r = random.randint(0, len(env.agents)-1)
                        a = env.agents[r]
                        clone = a.clone()
                        clone.x=random.randint(0,env.grid_size//CELL_SIZE -1)
                        clone.y=random.randint(0,env.grid_size//CELL_SIZE -1)
                        env.add_agent(clone)
        

pygame.quit()
