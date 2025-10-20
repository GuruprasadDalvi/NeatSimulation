
from pygame import draw, Rect
from pygame import gfxdraw
from pygame.font import Font

class Env:
    def __init__(self,wid,hei,cell_size,screen) -> None:
        self.wid = wid
        self.hei = hei
        self.cell_size = cell_size
        self.screen = screen
        self.grid = [[0]*(wid//cell_size)]*(hei//cell_size)
        self.sound_grid = [[0]*(wid//cell_size)]*(hei//cell_size)
        self.border_color = [10,10,10]
        self.bg_color = [44,44,44]
        self.font = Font(None, 16)
        self.selected = [0,0]
    
    def select(self,x,y):
        self.selected = [x//self.cell_size,y//self.cell_size]
        
        
    def render(self):
        
        for x in range(0, self.wid, self.cell_size):
            for y in range(0, self.hei, self.cell_size):
                rect = Rect(x, y, self.cell_size, self.cell_size)
                draw.rect(self.screen, self.border_color, rect, 1)
                
                # Render the x, y coordinates on the cell
                if x//self.cell_size==self.selected[0] and  y//self.cell_size==self.selected[1]:
                    text = self.font.render(f"({x // self.cell_size}, {y // self.cell_size})", True, (255, 255, 255))
                    text_rect = text.get_rect()
                    text_rect.center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    self.screen.blit(text, text_rect)
                
    def update(self):
        pass

