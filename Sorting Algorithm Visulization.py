import pygame
import math
import random
# Initial all the Pygame Modules
pygame.init()
# Graw images class
class DrawInformation:
    # set color
    BLACK = 0,0,0
    WHITE = 255,255,255
    RED = 255,0,0
    GREEN = 0,255,0
    BACKGROUND_COLOR = WHITE

    # Set bar grey color
    GRADIENTS=[(128,128,128),(160,160,160),(192,192,192)]
    #Set Font type  and size (the height of the font)
    FONT = pygame.font.SysFont('comicsans',30)
    LARGE_FONT = pygame.font.SysFont('cosmics',40)
    # Set side pad and top pad
    SIDE_PAD = 100
    TOP_PAD = 150

    def __init__(self,width, height,lst):
        self.width = width
        self.height = height
        self.set_list(lst)
        # pygame module to control the display window and screen
        # set_mode(size=(0, 0), flags=0, depth=0, display=0, vsync=0) -> Surface
        # This function will create a display Surface.
        # The arguments passed in are requests for a display type.
        # The actual created display will be the best possible match supported by the system.
        self.window  = pygame.display.set_mode((width,height))
        pygame.display.set_caption("Sorting Algorithm Visulization")
    # Set the list and create the bar based on the list
    def set_list(self,lst):
        self.lst = lst
        self.min_val = min(lst)
        self.max_val = max(lst)

        self.block_width = round((self.width - self.SIDE_PAD)/len(lst))
        self.block_height = math.floor((self.height-self.TOP_PAD)/(self.max_val-self.min_val))
        self.start_x = self.SIDE_PAD//2
def draw(draw_info,algo_name,ascending):
    #draw_info.window create a surface and the fill() function
    #set the background color for it.
    draw_info.window.fill(draw_info.BACKGROUND_COLOR)
    # draw text on a new Surface
    # render(text, antialias, color, background=None) -> Surface
    title = draw_info.LARGE_FONT.render(f"{algo_name} - {'Ascending' if ascending else 'Descending'}",1,draw_info.GREEN)
    # draw one image onto another
    # blit(source, dest, area=None, special_flags=0) -> Rect
    draw_info.window.blit(title,(draw_info.width/2-title.get_width()/2,5))

    controls = draw_info.FONT.render("R -Reset | SPACE - Start Sorting | A - Ascending | D - Descending",1,draw_info.BLACK)
    draw_info.window.blit(controls,(draw_info.width/2-controls.get_width()/2,45))

    sorting = draw_info.FONT.render("I - Insertion Sort | B - Bubble Sort | S - Selection Sort",1,draw_info.BLACK)
    draw_info.window.blit(sorting,(draw_info.width/2-sorting.get_width()/2,75))

    draw_list(draw_info)
    # Update the portions of the screen
    # update(rectangle=None) -> None
    # update(rectangle_list) -> None
    # If no argument is passed it updates the entire Surface area
    pygame.display.update()

def draw_list(draw_info,color_positions = {},clear_bg = False):
    lst = draw_info.lst
    if clear_bg:
        # pygame object for storing rectangular coordinates
        # Rect(left, top, width, height) -> Rect
        clear_rect = (draw_info.SIDE_PAD//2, draw_info.TOP_PAD,draw_info.width-draw_info.SIDE_PAD, draw_info.height - draw_info.TOP_PAD)
        # Draws a rectangle on the given surface.
        # rect(surface, color, rect) -> Rect
        pygame.draw.rect(draw_info.window,draw_info.BACKGOUND_COLOR,clear_rect)
    # Draw Block (Bar) for each item in the list
    for i,val in enumerate(lst):
        x = draw_info.start_x+i*draw_info.block_width
        y = draw_info.height - (val-draw_info.min_val)*draw_info.block_height

        color = draw_info.GRADIENTS[i%3]
        if i in color_positions:
            color = color_positions[i]
        pygame.draw.rect(draw_info.window,color,(x,y,draw_info.block_width,draw_info.block_height))

    if clear_bg:
        pygame.display.update()
def generate_starting_list(n,min_val,max_val):
    lst = []
    for _ in range(n):
        val = random.randint(min_val,max_val)
        lst.append(val)
    return lst
def buble_sort(draw_info,ascending = True):
    lst = draw_info.lst
    for i in range(len(lst)-1):
        for j in range(len(lst)-1-i):
            if (lst[j]>lst[j+1] and ascending) or (lst[j]<lst[j+1] and not ascending):
                lst[j],lst[j+1] = lst[j+1],lst[j]
                draw_list(draw_info,{j:draw_info.GREEN,j+1:draw_info.RED},True)
                yield True
    return lst
def insertion_sort(draw_info,ascending = True):
    lst = draw_info.lst
    for i in range(1,len(lst)):
        current  = lst[i]
        while True:
            ascending_sort = i>0 and lst[i-1]>current and ascending
            descending_sort = i>0 and lst[i-1]<current and not ascending
            if not ascending_sort and not descending_sort:
                break
            lst[i] = lst[i-1]
            i=i-1
            lst[i]=current
            draw_list(draw_info,{i-1:draw_info.GREEN,i:draw_info.RED},True)
            yield True
    return lst

def main():
    run = True
    # create an object to help track time
    clock = pygame.time.Clock()
    n = 50
    min_val = 0
    max_val = 100
    lst = generate_starting_list(n,min_val,max_val)
    draw_info = DrawInformation(800, 600, lst)
    sorting = False
    ascending = True
    sorting_algorithm = buble_sort
    sorting_algo_name = "Bubble Sort"
    sorting_algorithm_generator = None

    while run:
        # This method should be called once per frame. It will compute how many milliseconds have passed since the previous call.
        # If you pass the optional framerate argument the function will delay to keep the game running slower
        # than the given ticks per second. This can be used to help limit the runtime speed of a game.
        # By calling Clock.tick(40) once per frame, the program will never run at more than 40 frames per second.
        clock.tick(60)
        if sorting:
            try:
                next(sorting_algorithm_generator)
            except StopIteration:
                sorting = False
        else:
            draw(draw_info,sorting_algo_name,ascending)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type!= pygame.KEYDOWN:
                continue
            if event.key==pygame.K_r:
                lst = generate_starting_list(n,min_val,max_val)
                draw_info.set_list(lst)
                sorting = False
            elif event.key==pygame.K_SPACE and sorting == False:
                sorting = True
                sorting_algorithm_generator= sorting_algorithm(draw_info,ascending)
            elif event.key==pygame.K_a and not sorting:
                ascending = True
            elif event.key==pygame.K_d and not sorting:
                ascending = False
            elif event.key==pygame.K_i and not sorting:
                sorting_algorithm = insertion_sort
                sorting_algo_name = "Insertion Sort"
            elif event.key==pygame.K_b and not sorting:
                sorting_algorithm = buble_sort
                sorting_algo_name = "Buble Sort"
    pygame.quit()
if __name__ =="_main_":
    main()

