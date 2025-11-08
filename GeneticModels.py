import random
import base64
from pygame import draw, Rect
from pygame import gfxdraw
from NEAT import Network
from pygame.font import Font
import copy
import time
import pygame

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
        # Sound grid (communication field), separate from objects grid
        self.sound_grid = [[0.0 for _ in range((self.grid_size//cell_size))] for _ in range((self.grid_size//cell_size))]
        # self.food_grid = [[0]*(wid//cell_size)]*(hei//cell_size)
        self.border_color = [10,10,10]
        self.temp = 0
        self.bg_color  = [10,10,50]
        self.selected = None
        self.font = Font(None, 16)
        self.agents = []
        self.foods = []
        self.predators = []
        self.increasing = True
        self.maxAge = 0
        self.maxFitness = 0
        self.maxTraveled = 0
        self.maxConsumed = 0
        self.maxChild = 0
        self.maxAttacked = 0
        self.maxPredatorAttacksReceived = 0
        self.maxPredatorHits = 0
        self.maxGeneration = 0
        self.age = 0
        self.start_time = time.time()
        # Keep recent population counts for on-screen chart
        self.population_history = []
        # Keep recent available food counts for on-screen chart
        self.food_history = []
        # Simulation control
        self.paused = False
        self._request_step = False
		# Fitness weight UI state
        self.fitness_weight_names = [
			"Age", "Traveled", "Consumed", "Children",
			"Attacks", "Predator Hits Recv", "Predator Hits", "Generation"
		]
		# initialize equal weights that sum to 1
        eq = 1.0 / 8.0
        self.fitness_weights = [eq for _ in range(8)]
        self._active_weight_slider = None  # index while dragging, or None
        self._weight_slider_rects = []  # updated during render for hit-testing
        
        #CSV file for max age
        with open("max_fitness.csv", "w") as f:
            f.write("size,generation,attacked,predator_attacks_received,predator_hits,consumed,child,age,traveled,fitness, population\n")
        
    def select(self,x,y):
        x=x//self.cell_size
        y=y//self.cell_size
        if x>=0 and x<len(self.grid) and y>=0 and y<len(self.grid[0]) and type(self.grid[x][y])==Agent:
            self.selected = self.grid[x][y]
        
        print(f"Selected {self.grid[x][y].objId}  {self.selected}")
        print(f"North To selected: {self.getNorthCellId(x,y)}")
        print(f"East To selected: {self.getEastCellId(x,y)}")
        print(f"South To selected: {self.getSouthCellId(x,y)}")
        print(f"West To selected: {self.getWestCellId(x,y)}")
        print(f"NorthEast To selected: {self.getNorthEastCellId(x,y)}")
        print(f"SouthEast To selected: {self.getSouthEastCellId(x,y)}")
        print(f"SouthWest To selected: {self.getSouthWestCellId(x,y)}")
        print(f"NorthWest To selected: {self.getNorthWestCellId(x,y)}")
        
    def render(self):
        # Update agents only if running or a single step is requested
        if not self.paused or self._request_step:
            for a in list(self.agents):
                a.update()
                if a.traveled>self.maxTraveled:
                    self.maxTraveled = a.traveled
                if a.consumed>self.maxConsumed:
                    self.maxConsumed = a.consumed
                if a.child>self.maxChild:
                    self.maxChild = a.child
                if a.attacked>self.maxAttacked:
                    self.maxAttacked = a.attacked
                if a.predator_attacks_received>self.maxPredatorAttacksReceived:
                    self.maxPredatorAttacksReceived = a.predator_attacks_received
                if a.predator_hits>self.maxPredatorHits:
                    self.maxPredatorHits = a.predator_hits
                if a.generation>self.maxGeneration:
                    self.maxGeneration = a.generation
                if a.age>self.maxAge:
                    self.maxAge = a.age
                if a.get_fitness()>self.maxFitness:
                    self.maxFitness = a.get_fitness()
                    if a!=self.selected:
                        #write to csv
                        with open("max_fitness.csv", "a") as f:
                            f.write(f"{a.get_csv()}, {len(self.agents)}\n")
                    self.selected = a
            # Update predators
            for p in list(self.predators):
                p.update()
            self.age+=1
            # clear step request after processing one tick
            self._request_step = False
        # update elapsed real time (seconds) only while running
        if not self.paused:
            try:
                self.elapsed_seconds = max(0.0, time.time() - self.start_time)
            except Exception:
                self.elapsed_seconds = 0.0
        # track population history (cap to recent N points)
        try:
            self.population_history.append(len(self.agents))
            if len(self.population_history) > 600:
                self.population_history = self.population_history[-600:]
        except Exception:
            pass
        # track food history (cap to recent N points)
        try:
            self.food_history.append(len(self.foods))
            if len(self.food_history) > 600:
                self.food_history = self.food_history[-600:]
        except Exception:
            pass
        for x in range(0, self.grid_size, self.cell_size):
            for y in range(0, self.grid_size, self.cell_size):
                a = self.grid[y//self.cell_size][x//self.cell_size]
                a.render()

        # If an agent is selected, highlight all visited cells it has traveled
        try:
            if self.selected and type(self.selected) == Agent and getattr(self.selected, 'visited', None):
                cell = self.cell_size
                overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                # warm yellow translucent fill
                overlay.fill((255, 230, 120, 80))
                for (vx, vy) in self.selected.visited:
                    # bounds-safe blit; out-of-range entries are ignored
                    if 0 <= vx < (self.grid_size // cell) and 0 <= vy < (self.grid_size // cell):
                        self.screen.blit(overlay, (vx * cell, vy * cell))
        except Exception:
            pass
                
                # Render the x, y coordinates on the cell
        # Right-side UI layout: Stats (if selected), Population chart, Brain (if selected)
        panel_left = self.grid_size + 10
        panel_top = 10
        panel_width = max(0, self.wid - panel_left - 10)
        y_cursor = panel_top
        # Fitness Weights panel
        if panel_width > 10:
            weights_panel_h = self._render_fitness_weights_panel(panel_left, panel_width, y_cursor)
            y_cursor += weights_panel_h + 10
		# Stats panel if an agent is selected
        if self.selected and type(self.selected)==Agent:
            a = self.selected
            panel_height = 200
            if panel_width > 10:
                stats_rect = Rect(panel_left, y_cursor, panel_width, panel_height)
                draw.rect(self.screen, [25,25,40], stats_rect, border_radius=6)

            texts = [
                f"Attack:    {a.attacked}, Traveled:  {round(a.traveled,4)}",
                f"Predator Hits Received: {a.predator_attacks_received}",
                f"Predator Hits: {a.predator_hits}",
                f"Consumed:  {a.consumed}",
                f"Age:       {round(a.age,4)}, Children:  {round(a.child,4)}",
                f"Energy:{round(a.energy,1)}, Fitness: {round(a.get_fitness(),4)}",
                f"Generation: {a.generation} Size: {a.brain.get_size()}",
                f"X: {round(a.x)}, Y: {round(a.y)}"
            ]
            yPos = y_cursor + 15
            for t in texts:
                text = self.font.render(t, True, (255, 255, 255))
                text_rect = text.get_rect()
                text_rect.center = (int(self.wid*.9), yPos)
                self.screen.blit(text, text_rect)
                yPos+=25
            y_cursor += panel_height + 10
        # Population chart panel
        pop_height = 140
        if panel_width >= 40:
            self._draw_population_chart(panel_left, panel_width, y_cursor, pop_height)
            y_cursor += pop_height + 10
        # Brain panel for selected agent
        if self.selected and type(self.selected)==Agent:
            self._draw_brain(self.selected, y_cursor)
        
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

        # Decay sound field towards 0 each tick
        decay = 0.9
        threshold = 1e-3
        sg = self.sound_grid
        for i in range(len(sg)):
            row = sg[i]
            for j in range(len(row)):
                v = row[j] * decay
                if -threshold < v < threshold:
                    v = 0.0
                row[j] = v
        
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
    
    def damage(self,x,y,amount, attacker):
        try:
            a = self.grid[x][y]
            if type(a)==Agent:
                a.energy -=amount
                if(a.energy<0):
                    self.grid[a.x][a.y] = DefaultObj(self,a.x, a.y)
                    self.agents.remove(a)
                    if self.selected == a:
                        self.selected = None 
            elif type(a)==Predator:
                a.energy -= amount
                attacker.predator_hits += 1
                if(a.energy<0):
                    self.grid[a.x][a.y] = DefaultObj(self,a.x, a.y)
                    self.predators.remove(a)
                    if self.selected == a:
                        self.selected = None 
        except:
            pass
    
    def predator_damage(self,x,y,amount):
        try:
            a = self.grid[x][y]
            if type(a)==Agent:
                a.energy -= amount
                try:
                    a.predator_attacks_received += 1
                except Exception:
                    pass
                if a.energy < 0:
                    self.grid[a.x][a.y] = DefaultObj(self,a.x, a.y)
                    if a in self.agents:
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
        else:
            return 

    def add_food(self, food):
        if type(self.grid[food.x][food.y]) == DefaultObj:
            self.grid[food.x][food.y]= food
            self.foods.append(food)

    def add_predator(self, predator):
        if type(self.grid[predator.x][predator.y]) == DefaultObj:
            self.grid[predator.x][predator.y] = predator
            self.predators.append(predator)

    # ===== Sound helpers =====
    def add_sound(self, x, y, value):
        try:
            self.sound_grid[x][y] = self.sound_grid[x][y] + float(value)
        except Exception:
            # out of bounds are ignored
            pass

    def get_sound_9(self, x, y):
        # Order: N, NE, E, SE, S, SW, W, NW, C
        out = []
        def get(xx, yy):
            try:
                return float(self.sound_grid[xx][yy])
            except Exception:
                return 0.0
        out.append(get(x, y-1))   # N
        out.append(get(x+1, y-1)) # NE
        out.append(get(x+1, y))   # E
        out.append(get(x+1, y+1)) # SE
        out.append(get(x, y+1))   # S
        out.append(get(x-1, y+1)) # SW
        out.append(get(x-1, y))   # W
        out.append(get(x-1, y-1)) # NW
        out.append(get(x, y))     # C
        return out

    def _draw_brain(self, agent: 'Agent', panel_top):
        # Panel bounds on the right side
        panel_left = self.grid_size + 10
        panel_right = self.wid - 10
        panel_bottom = self.hei - 20
        panel_width = max(0, panel_right - panel_left)
        panel_height = max(0, panel_bottom - panel_top)
        if panel_width < 20 or panel_height < 20:
            return

        # Draw panel background
        bg_rect = Rect(panel_left, panel_top, panel_width, panel_height)
        draw.rect(self.screen, [25,25,40], bg_rect, border_radius=6)

        # Validate brain
        brain = agent.brain
        # Prefer NEAT.Network built-in renderer if available
        if hasattr(brain, 'render'):
            try:
                brain.render(self.screen, (panel_left, panel_top), (panel_width, panel_height))
                return
            except Exception:
                pass
        layers = getattr(brain, 'layers', None)
        weights = getattr(brain, 'weights', None)
        activations = getattr(brain, 'activisions', None)
        if not layers or not weights:
            return

        layer_count = len(layers)
        if layer_count < 2:
            return

        # Layout: x positions per layer
        dx = panel_width / (layer_count + 1)
        layer_x_positions = [int(panel_left + (i+1)*dx) for i in range(layer_count)]

        # Precompute node positions for each layer
        node_positions = []  # list of lists of (x,y)
        for idx, node_count in enumerate(layers):
            if node_count <= 0:
                node_positions.append([])
                continue
            dy = panel_height / (node_count + 1)
            x = layer_x_positions[idx]
            positions = [(x, int(panel_top + (j+1)*dy)) for j in range(node_count)]
            node_positions.append(positions)

        # Helper to map values to colors
        def color_for_weight(w):
            # Red for positive, Blue for negative, intensity by magnitude
            m = abs(float(w))
            # squash magnitude to [0,1]
            m = m / (1.0 + m)
            val = int(40 + 215*m)
            if w >= 0:
                return (val, 40, 40)
            else:
                return (40, 40, val)

        def color_for_activation(a):
            # Map [-1,1] -> blue(neg) to red(pos), grey near 0
            try:
                v = max(-1.0, min(1.0, float(a)))
            except Exception:
                v = 0.0
            if v >= 0:
                intensity = int(60 + 195*v)
                return (intensity, 60, 60)
            else:
                intensity = int(60 + 195*(-v))
                return (60, 60, intensity)

        # Draw edges between layers
        for l in range(layer_count - 1):
            w_mat = weights[l]
            from_positions = node_positions[l]
            to_positions = node_positions[l+1]
            if not from_positions or not to_positions:
                continue
            rows = len(w_mat)
            cols = len(w_mat[0]) if rows>0 else 0
            for j in range(rows):  # to-node index
                for i in range(cols):  # from-node index
                    w = w_mat[j][i]
                    c = color_for_weight(w)
                    th = 1 if abs(w) < 0.5 else (2 if abs(w) < 1.0 else 3)
                    (x1,y1) = from_positions[i]
                    (x2,y2) = to_positions[j]
                    draw.line(self.screen, c, (x1,y1), (x2,y2), th)

        # Draw nodes with activation coloring
        radius = 7
        for l in range(layer_count):
            acts = None
            if activations and l < len(activations):
                acts = activations[l]
            for idx, (x,y) in enumerate(node_positions[l]):
                val = 0
                try:
                    if acts is not None:
                        val = acts[idx]
                except Exception:
                    val = 0
                col = color_for_activation(val)
                gfxdraw.filled_circle(self.screen, x, y, radius, col)
                gfxdraw.aacircle(self.screen, x, y, radius, (220,220,220))

        # Titles
        title = self.font.render("Brain", True, (255,255,255))
        title_rect = title.get_rect()
        title_rect.center = (int((panel_left + panel_right)/2), panel_top - 12)
        self.screen.blit(title, title_rect)

    def _draw_population_chart(self, left, chart_w, top, chart_h):
        hist = self.population_history
        food_hist = self.food_history
        if (not hist and not food_hist) or chart_w <= 0 or chart_h <= 0:
            return
        bg_rect = Rect(left, top, chart_w, chart_h)
        # Background panel
        draw.rect(self.screen, [25,25,40], bg_rect, border_radius=6)
        # Title with current population
        try:
            last_pop = hist[-1] if hist else 0
            last_food = food_hist[-1] if food_hist else 0
            title = self.font.render(f"Population {last_pop}  |  Food {last_food}", True, (255,255,255))
            self.screen.blit(title, (left + 10, top + 8))
        except Exception:
            pass
        # Plot area
        pad_left, pad_right, pad_top, pad_bottom = 8, 8, 26, 8
        plot_left = left + pad_left
        plot_top = top + pad_top
        plot_w = max(10, chart_w - pad_left - pad_right)
        plot_h = max(10, chart_h - pad_top - pad_bottom)
        draw.rect(self.screen, (35,35,55), Rect(plot_left, plot_top, plot_w, plot_h), 1)
        # Choose visible windows
        window_p = hist[-min(len(hist), 300):] if hist else []
        window_f = food_hist[-min(len(food_hist), 300):] if food_hist else []
        if len(window_p) < 2 and len(window_f) < 2:
            return
        try:
            values = []
            if window_p:
                values.extend(window_p)
            if window_f:
                values.extend(window_f)
            min_v = min(values)
            max_v = max(values)
            if max_v == min_v:
                max_v = min_v + 1
            # Build polyline for population (light blue)
            if window_p and len(window_p) >= 2:
                n_p = len(window_p)
                pts_p = []
                for i, val in enumerate(window_p):
                    x = plot_left + int(i * (plot_w - 1) / (n_p - 1))
                    norm = (val - min_v) / (max_v - min_v)
                    y = plot_top + int((1.0 - norm) * (plot_h - 1))
                    pts_p.append((x, y))
                if len(pts_p) >= 2:
                    draw.lines(self.screen, (120, 200, 255), False, pts_p, 2)
            # Build polyline for food (green)
            if window_f and len(window_f) >= 2:
                n_f = len(window_f)
                pts_f = []
                for i, val in enumerate(window_f):
                    x = plot_left + int(i * (plot_w - 1) / (n_f - 1))
                    norm = (val - min_v) / (max_v - min_v)
                    y = plot_top + int((1.0 - norm) * (plot_h - 1))
                    pts_f.append((x, y))
                if len(pts_f) >= 2:
                    draw.lines(self.screen, (120, 255, 120), False, pts_f, 2)
            # Min/Max labels
            max_txt = self.font.render(str(max_v), True, (180,180,200))
            min_txt = self.font.render(str(min_v), True, (180,180,200))
            self.screen.blit(max_txt, (left + chart_w - 50, plot_top - 16))
            self.screen.blit(min_txt, (left + chart_w - 50, plot_top + plot_h - 14))
            # Legend
            try:
                legend_pop = self.font.render("Pop", True, (120,200,255))
                legend_food = self.font.render("Food", True, (120,255,120))
                self.screen.blit(legend_pop, (left + chart_w - 110, top + 8))
                self.screen.blit(legend_food, (left + chart_w - 60, top + 8))
            except Exception:
                pass
        except Exception:
            pass

    def _render_fitness_weights_panel(self, left, panel_w, top):
        # Layout constants
        padding = 10
        title_h = 22
        row_h = 26
        label_w = 150
        track_h = 8
        knob_r = 6
        n = len(self.fitness_weight_names)
        panel_h = title_h + padding + n * row_h + padding
        # Background
        bg_rect = Rect(left, top, panel_w, panel_h)
        draw.rect(self.screen, [25,25,40], bg_rect, border_radius=6)
        # Title and sum
        try:
            title = self.font.render("Fitness Weights", True, (255,255,255))
            self.screen.blit(title, (left + padding, top + 4))
            total = sum(self.fitness_weights) if self.fitness_weights else 0.0
            col = (120,255,120) if abs(total - 1.0) < 1e-3 else (255,180,120)
            sum_txt = self.font.render(f"Sum: {total:.2f}", True, col)
            self.screen.blit(sum_txt, (left + panel_w - padding - 80, top + 4))
        except Exception:
            pass
        # Sliders
        slider_left = left + padding + label_w + 10
        slider_w = max(60, panel_w - (slider_left - left) - padding - 40)
        self._weight_slider_rects = []
        for i, name in enumerate(self.fitness_weight_names):
            y = top + title_h + padding + i * row_h
            # Label
            try:
                label = self.font.render(name, True, (220,220,240))
                self.screen.blit(label, (left + padding, y - 2))
            except Exception:
                pass
            # Track
            track_rect = Rect(slider_left, y + (row_h - track_h)//2, slider_w, track_h)
            draw.rect(self.screen, (50,60,80), track_rect, border_radius=3)
            # Knob at value
            val = 0.0
            try:
                val = max(0.0, min(1.0, float(self.fitness_weights[i])))
            except Exception:
                val = 0.0
            kx = int(track_rect.left + val * track_rect.width)
            ky = track_rect.centery
            try:
                gfxdraw.filled_circle(self.screen, kx, ky, knob_r, (180,200,255))
                gfxdraw.aacircle(self.screen, kx, ky, knob_r, (235,235,245))
            except Exception:
                pass
            # Current value text
            try:
                val_txt = self.font.render(f"{val:.2f}", True, (200,200,210))
                self.screen.blit(val_txt, (track_rect.right + 8, y - 2))
            except Exception:
                pass
            # Larger hit rect for easier clicking
            hit_rect = Rect(track_rect.left - 8, y, track_rect.width + 16, row_h)
            self._weight_slider_rects.append(hit_rect)
        return panel_h

    def _rebalance_weights(self, fixed_index):
        # Keep the fixed index value; scale others to keep total exactly 1.0
        try:
            n = len(self.fitness_weights)
            if n <= 0:
                return
            v_fixed = max(0.0, min(1.0, float(self.fitness_weights[fixed_index])))
            other_sum = sum(self.fitness_weights[j] for j in range(n) if j != fixed_index)
            target_other = max(0.0, 1.0 - v_fixed)
            if other_sum <= 1e-9:
                share = target_other / max(1, n - 1)
                for j in range(n):
                    if j == fixed_index:
                        self.fitness_weights[j] = v_fixed
                    else:
                        self.fitness_weights[j] = share
                return
            scale = target_other / other_sum
            for j in range(n):
                if j == fixed_index:
                    self.fitness_weights[j] = v_fixed
                else:
                    self.fitness_weights[j] = max(0.0, min(1.0, float(self.fitness_weights[j]) * scale))
        except Exception:
            pass

    def _update_weight_from_mouse(self, index, mouse_x):
        try:
            if index < 0 or index >= len(self._weight_slider_rects):
                return
            track_rect = self._weight_slider_rects[index]
            # deduce the inner track from hit rect: shrink margins used above
            inner_left = track_rect.left + 8
            inner_right = track_rect.right - 8
            inner_w = max(1, inner_right - inner_left)
            raw = (mouse_x - inner_left) / float(inner_w)
            val = 0.0 if raw < 0.0 else (1.0 if raw > 1.0 else raw)
            self.fitness_weights[index] = val
            self._rebalance_weights(index)
            self.recalculate_max_fitness()
        except Exception:
            pass
    def recalculate_max_fitness(self):
        self.maxFitness = 0
        for a in self.agents:
            if a.get_fitness()>self.maxFitness:
                self.maxFitness = a.get_fitness()
    def handle_event(self, event):
        try:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Grid selection
                if x < self.grid_size:
                    self.select(x, y)
                # Slider interaction
                for i, r in enumerate(self._weight_slider_rects):
                    try:
                        if r.collidepoint(x, y):
                            self._active_weight_slider = i
                            self._update_weight_from_mouse(i, x)
                            break
                    except Exception:
                        pass
            elif event.type == pygame.MOUSEMOTION:
                if self._active_weight_slider is not None:
                    if pygame.mouse.get_pressed()[0]:
                        x, _ = pygame.mouse.get_pos()
                        self._update_weight_from_mouse(self._active_weight_slider, x)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._active_weight_slider = None
        except Exception:
            pass

class DefaultObj:
    def __init__(self, env: Environment,x,y) -> None:
        self.env = env
        self.objId = -1.0
        self.x = x
        self.y = y
    
    def render(self):
        # Subtle checker pattern for background cells
        px = self.x*self.env.cell_size
        py = self.y*self.env.cell_size
        rect = Rect(px, py, self.env.cell_size, self.env.cell_size)
        # base tile color
        ix = (self.x + self.y) % 2
        base_col = [18,18,36] if ix == 0 else [22,22,44]
        draw.rect(self.env.screen, base_col, rect)
        # grid line
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
        # Inputs: 12 existing + 9 sound inputs = 21
        # Outputs: 6 existing + 9 sound outputs = 15
        self.brain = Network(21, 15)
        self.energy = 2000
        self.maxenergy = 5000
        self.attacked = 0
        self.predator_attacks_received = 0
        self.predator_hits = 0
        self.consumed = 0
        self.child = 0
        self.age = 0
        self.traveled = 0
        self.generation = 0
        # Track all visited grid cells for highlighting when selected
        self.visited = set()
        self.visited.add((self.x, self.y))
  
    def get_fitness(self):
        def safe_div(numerator, denominator):
            return (numerator / denominator) if denominator not in (0, None) else 0.0
        metrics = [
            safe_div(self.age, self.env.maxAge),
            safe_div(self.traveled, self.env.maxTraveled),
            safe_div(self.consumed, self.env.maxConsumed),
            safe_div(self.child, self.env.maxChild),
            safe_div(self.attacked, self.env.maxAttacked),
            safe_div(self.predator_attacks_received, self.env.maxPredatorAttacksReceived),
            safe_div(self.predator_hits, self.env.maxPredatorHits),
            safe_div(self.generation, self.env.maxGeneration)
        ]
        try:
            weights = getattr(self.env, "fitness_weights", None)
            if not weights or len(weights) != len(metrics):
                weights = [1.0/len(metrics) for _ in metrics]
            total = sum(weights)
            if total <= 1e-12:
                total = 1.0
                weights = [1.0/len(metrics) for _ in metrics]
            norm_w = [w/total for w in weights]
            return sum(m*w for m, w in zip(metrics, norm_w))
        except Exception:
            return sum(metrics) * (1.0 / 8.0)
    def update(self):
        self.age+=1
        base_inputs = [
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
        ]
        sound_inputs = self.env.get_sound_9(self.x, self.y)
        preds = self.brain.activate(base_inputs + sound_inputs)
        
        # Attack
        if float(preds[2])>0:
            self.attack()
        # Reproduce 
        if(self.energy>2000) and preds[3]>0:
            self.reproduce()
            self.energy-=200
        
        # Eat
        if(float(preds[4]))>0:
            self.eat()
            if self.energy>self.maxenergy:
                self.energy = self.maxenergy
        
        if(self.energy<0):
            self.env.grid[self.x][self.y] = DefaultObj(self.env,self.x, self.y)
            if self in self.env.agents:
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
        self.energy-=self.brain.get_size()/10000

        # Emit sound to 9 surrounding cells (N, NE, E, SE, S, SW, W, NW, C)
        try:
            sound_ops = [preds[i] for i in range(6, 15)]
            offsets = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,0)]
            for (dx, dy), val in zip(offsets, sound_ops):
                self.env.add_sound(self.x+dx, self.y+dy, float(val))
        except Exception:
            pass
        
        
        # pass

    def move(self,dx,dy):
        if dx==0 and dy==0:
            self.energy-=15
            return
        if self.env.isCellEmpty(self.x+dx,self.y+dy):
            df = DefaultObj(self.env, self.x,self.y)
            newX = (self.x + dx)%len(self.env.grid[self.x])
            newY = (self.y + dy)%len(self.env.grid[self.y])
            self.env.grid[newX][newY] = self
            self.env.grid[self.x][self.y] = df
            # record current and next cell as visited
            self.visited.add((self.x, self.y))
            self.x= newX
            self.y= newY
            self.visited.add((self.x, self.y))
            self.traveled+=1
        
    def reproduce(self):
        newX = random.randint(0,self.env.grid_size//self.env.cell_size -1)
        newY = random.randint(0,self.env.grid_size//self.env.cell_size -1)
        a = Agent(self.env,self.screen)
        a.x = newX
        a.y = newY
        a.brain = self.brain.clone()
        a.color = self.color
        if random.random()<0.95: 
            a.brain.mutate()
            # Change Color slightly
            a.color = [a.color[0] + random.randint(-10,10),a.color[1] + random.randint(-10,10),a.color[2] + random.randint(-10,10)]
        a.generation = self.generation+1
        self.env.add_agent(a)
        self.child+=1
        
    def clone(self):
        a = Agent(self.env,self.screen)
        a.x = self.x
        a.y = self.y
        a.color = self.color
        a.brain = self.brain.clone()
        if random.random()<0.75:
            a.brain.mutate()
            # Change Color slightly
            a.color = [a.color[0] + random.randint(-10,10),a.color[1] + random.randint(-10,10),a.color[2] + random.randint(-10,10)]
        return a
    
    def attack(self):
            self.energy-=80
            self.env.damage(self.x,self.y-1,20,self)
            self.env.damage(self.x,self.y+1,20,self)
            
            self.env.damage(self.x-1,self.y,10,self)
            self.env.damage(self.x+1,self.y,10,self)
            
            self.env.damage(self.x-1,self.y-1,5,self)
            self.env.damage(self.x+1,self.y+1,5,self)
            self.env.damage(self.x+1,self.y-1,5,self)
            self.env.damage(self.x-1,self.y+1,5,self)
            self.attacked+=1

    def render(self):
        # Pretty agent rendering: circular body with subtle gradient, outline, selection ring and energy bar
        cell_size = self.env.cell_size
        px = self.x * cell_size
        py = self.y * cell_size
        cx = px + cell_size // 2
        cy = py + cell_size // 2
        radius = max(3, int(cell_size * 0.42))

        def clamp_channel(v):
            return 0 if v < 0 else (255 if v > 255 else int(v))

        def lighten(col, amt):
            return (
                clamp_channel(col[0] + amt),
                clamp_channel(col[1] + amt),
                clamp_channel(col[2] + amt),
            )

        def darken(col, amt):
            return (
                clamp_channel(col[0] - amt),
                clamp_channel(col[1] - amt),
                clamp_channel(col[2] - amt),
            )

        body_color = tuple(self.color)
        outer_color = darken(body_color, 20)
        inner_color = lighten(body_color, 35)

        # Simple radial gradient using concentric circles
        steps = 3
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1.0
            col = (
                clamp_channel(int(outer_color[0] * (1 - t) + inner_color[0] * t)),
                clamp_channel(int(outer_color[1] * (1 - t) + inner_color[1] * t)),
                clamp_channel(int(outer_color[2] * (1 - t) + inner_color[2] * t)),
            )
            r = int(radius * (1.0 - 0.18 * i))
            gfxdraw.filled_circle(self.screen, cx, cy, r, col)

        # Crisp outline
        gfxdraw.aacircle(self.screen, cx, cy, radius, lighten(body_color, 50))
        gfxdraw.aacircle(self.screen, cx, cy, radius + 1, (15, 15, 20))

        # Selection pulse ring
        if self.env.selected == self:
            pulse = (self.env.age % 30) / 30.0
            ring_color = (255, 225, 120)
            ring_radius = radius + 2 + int(3 * pulse)
            gfxdraw.aacircle(self.screen, cx, cy, ring_radius, ring_color)
            if ring_radius - 1 > 0:
                gfxdraw.aacircle(self.screen, cx, cy, ring_radius - 1, ring_color)

        # Energy bar under the agent
        bar_w = cell_size - 6
        bar_h = 4
        bar_x = px + 3
        bar_y = py + cell_size - bar_h - 2
        # background
        draw.rect(self.screen, (35, 35, 50), Rect(bar_x, bar_y, bar_w, bar_h), border_radius=2)
        # fill
        ratio = 0.0 if self.maxenergy <= 0 else max(0.0, min(1.0, float(self.energy) / float(self.maxenergy)))
        fill_w = max(0, int(bar_w * ratio))
        if fill_w > 0:
            # Color from red (low) to green (high)
            fill_r = int(255 * (1 - ratio))
            fill_g = int(200 * ratio + 30)
            fill_b = 40
            draw.rect(self.screen, (fill_r, fill_g, fill_b), Rect(bar_x, bar_y, fill_w, bar_h), border_radius=2)
    
    def eat(self):
        # East
        try:
            if type(self.env.grid[self.x+1][self.y])==Food:
                self.energy+=self.env.grid[self.x+1][self.y].energy
                self.env.foods.remove(self.env.grid[self.x+1][self.y])
                self.env.grid[self.x+1][self.y] = DefaultObj(self.env,self.x+1, self.y)
                self.consumed+=1
                return
            # North
            if type(self.env.grid[self.x][self.y-1])==Food:
                self.energy+=self.env.grid[self.x][self.y-1].energy
                self.env.foods.remove(self.env.grid[self.x][self.y-1])
                self.env.grid[self.x][self.y-1] = DefaultObj(self.env,self.x, self.y-1)
                self.consumed+=1
                return
            
            # West
            if type(self.env.grid[self.x-1][self.y])==Food:
                self.energy+=self.env.grid[self.x-1][self.y].energy
                self.env.foods.remove(self.env.grid[self.x-1][self.y])
                self.env.grid[self.x-1][self.y] = DefaultObj(self.env,self.x-1, self.y)
                self.consumed+=1
                return
            
            # South
            if type(self.env.grid[self.x][self.y+1])==Food:
                self.energy+=self.env.grid[self.x][self.y+1].energy
                self.env.foods.remove(self.env.grid[self.x][self.y+1])
                self.env.grid[self.x][self.y+1] = DefaultObj(self.env,self.x, self.y+1)
                self.consumed+=1
                return
        except:
            pass
    def get_csv(self)->str:
        return f"{self.brain.get_size()},{self.generation},{self.attacked},{self.predator_attacks_received},{self.predator_hits} ,{self.consumed},{self.child},{self.age},{self.traveled},{self.get_fitness()}"
            
        
        
class Predator:
    
    def __init__(self, env, screen) -> None:
        self.x = random.randint(0,len(env.grid[0])-1)
        self.y = random.randint(0,len(env.grid)-1)
        self.env = env
        self.objId = 0.25
        self.screen = screen
        self.color = (220, 70, 70)
    
    def update(self):
        # Attack all adjacent cells with small damage
        try:
            for dx, dy in [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]:
                self.env.predator_damage(self.x+dx, self.y+dy, 5)
        except Exception:
            pass
        # Random simple movement (including staying still)
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
        dx, dy = random.choice(dirs)
        if dx != 0 or dy != 0:
            self.move(dx, dy)
    
    def move(self, dx, dy):
        try:
            if self.env.isCellEmpty(self.x+dx, self.y+dy):
                df = DefaultObj(self.env, self.x, self.y)
                newX = (self.x + dx) % len(self.env.grid[self.x])
                newY = (self.y + dy) % len(self.env.grid[self.y])
                self.env.grid[newX][newY] = self
                self.env.grid[self.x][self.y] = df
                self.x = newX
                self.y = newY
        except Exception:
            pass
    
    def render(self):
        # Render as a red triangle
        cell = self.env.cell_size
        px = self.x * cell
        py = self.y * cell
        cx = px + cell // 2
        cy = py + cell // 2
        r = max(4, int(cell * 0.4))
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        try:
            draw.polygon(self.env.screen, (220,70,70), points)
            draw.polygon(self.env.screen, (30,30,30), points, 1)
        except Exception:
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
        # Food as glowing pellet
        cell = self.env.cell_size
        px = self.x*cell
        py = self.y*cell
        cx = px + cell//2
        cy = py + cell//2
        base = (90, 200, 255)
        core = (180, 240, 255)
        r = max(2, int(cell*0.28))
        # soft halo
        halo_r = r + 3
        draw.circle(self.screen, (20,40,60), (cx, cy), halo_r, 0)
        # core
        gfxdraw.filled_circle(self.screen, cx, cy, r, base)
        gfxdraw.filled_circle(self.screen, cx, cy, max(1, r-2), core)
        gfxdraw.aacircle(self.screen, cx, cy, r, (220, 240, 255))
        
    def update(self):
        pass