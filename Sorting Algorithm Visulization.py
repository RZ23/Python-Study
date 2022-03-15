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
    YELLOW = 0,0,255
    BACKGROUND_COLOR = WHITE

    # Set bar grey color
    # GRADIENTS=[(128,128,128),(160,160,160),(192,192,192)]
    GRADIENTS = [
        (128, 128, 128),
        (160, 160, 160),
        (192, 192, 192)
    ]
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

    sorting = draw_info.FONT.render("I - Insertion Sort | B - Bubble Sort | S - Selection Sort | M - Merge Sort | Q - Quick Sort ",1,draw_info.BLACK)
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
        pygame.draw.rect(draw_info.window,draw_info.BACKGROUND_COLOR,clear_rect)
    # Draw Block (Bar) for each item in the list
    for i,val in enumerate(lst):
        x = draw_info.start_x+i*draw_info.block_width
        y = draw_info.height - (val-draw_info.min_val)*draw_info.block_height

        color = draw_info.GRADIENTS[i % 3]
        if i in color_positions:
            color = color_positions[i]
        pygame.draw.rect(draw_info.window,color,(x,y,draw_info.block_width,draw_info.height))

    if clear_bg:
        pygame.display.update()
def generate_starting_list(n,min_val,max_val):
    lst = []
    for _ in range(n):
        val = random.randint(min_val,max_val)
        lst.append(val)
    return lst
def bubble_sort(draw_info,ascending = True):
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
def merge_sort_paten(draw_info,ascending = True):
    lst = draw_info.lst
    mid = len(lst)//2
    low = 0
    high = len(lst)-1
    draw_list(draw_info, {mid: draw_info.GREEN, low: draw_info.RED, high: draw_info.YELLOW}, True)
    draw_info.lst = merge_sort(lst,ascending)
    draw_list(draw_info,{mid:draw_info.GREEN,low:draw_info.RED,high:draw_info.YELLOW}, True)
    yield True
    return draw_info.lst
def merge_sort(lst,ascending):
    if len(lst)<=1:
        return lst
    mid = len(lst)//2
    left = merge_sort(lst[:mid],ascending)
    right = merge_sort(lst[mid:],ascending)
    sorted_lst = merge(left,right,ascending)
    return sorted_lst
def merge(left,right,ascending=True):
    sorted_list = []
    i = 0
    j= 0
    if ascending:
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                sorted_list.append(left[i])
                i=i+1
            else:
                sorted_list.append(right[j])
                j=j+1
        left_left = left[i:]
        right_left = right[j:]
        return sorted_list+left_left+right_left
    if not ascending:
        while i<len(left) and j<len(right):
            if left[i]>right[j]:
                sorted_list.append(left[i])
                i=i+1
            else:
                sorted_list.append(right[j])
                j=j+1
        left_left = left[i:]
        right_left = right[j:]
        return sorted_list+left_left+right_left
def selection_sort(draw_info,ascending = True):
    lst = draw_info.lst
    if ascending:
        for i in range(len(lst)-1):
            min = i
            min_val = lst[i]
            for j in range(i+1,len(lst)):
                if lst[j]<min_val:
                    min_val = lst[j]
                    min = j
            draw_list(draw_info, {i: draw_info.GREEN, j: draw_info.RED}, True)
            yield True
            if min!=i:
                lst[min],lst[i] = lst[i],lst[min]
            draw_list(draw_info, {i: draw_info.GREEN, min: draw_info.RED}, True)
            yield True
        return lst
    if not ascending:
        for i in range(len(lst)-1):
            max = i
            max_val = lst[i]
            for j in range(i+1, len(lst)):
                if lst[j] > max_val:
                    max_val = lst[j]
                    max = j
            draw_list(draw_info, {i: draw_info.GREEN, j: draw_info.RED}, True)
            yield True
            if max != i:
                lst[max], lst[i] = lst[i], lst[max]
            draw_list(draw_info, {i: draw_info.GREEN, max: draw_info.RED}, True)
            yield True
        return lst
def quick_sort_patern(draw_info,ascending = True):
    lst = draw_info.lst
    print(lst)
    draw_info.lst = quick_sort(lst,ascending)
    draw_list(draw_info,{},True)
    yield True
    print(draw_info.lst)
    return draw_info.lst

def quick_sort(lst,ascending = True):
    if len(lst)<=1:
        return lst
    pivot = lst[-1]
    left = []
    right = []
    if ascending:
        for i in range(len(lst)-1):
            if lst[i]<=pivot:
                left.append(lst[i])
            else:
                right.append(lst[i])
        return quick_sort(left,ascending)+[pivot]+quick_sort(right,ascending)
    else:
        for i in range(len(lst)-1):
            if lst[i]>pivot:
                left.append(lst[i])
            else:
                right.append(lst[i])
        return quick_sort(left,ascending)+[pivot]+quick_sort(right,ascending)

def main():
    run = True
    # create an object to help track time
    clock = pygame.time.Clock()
    n = 25
    min_val = 0
    max_val = 100
    lst = generate_starting_list(n,min_val,max_val)
    draw_info = DrawInformation(1500, 800, lst)
    sorting = False
    ascending = True
    sorting_algorithm = bubble_sort
    sorting_algo_name = "Bubble Sort"
    sorting_algorithm_generator = None

    while run:
        # This method should be called once per frame. It will compute how many milliseconds have passed since the previous call.
        # If you pass the optional framerate argument the function will delay to keep the game running slower
        # than the given ticks per second. This can be used to help limit the runtime speed of a game.
        # By calling Clock.tick(40) once per frame, the program will never run at more than 40 frames per second.
        clock.tick(120)
        if sorting:
            try:
                next(sorting_algorithm_generator)
            except StopIteration:
                sorting = False
        else:
            draw(draw_info, sorting_algo_name, ascending)

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
                sorting_algorithm = bubble_sort
                sorting_algo_name = "Buble Sort"
            elif event.key == pygame.K_s and not sorting:
                sorting_algorithm = selection_sort
                sorting_algo_name = "Selection Sort"
            elif event.key == pygame.K_m and not sorting:
                sorting_algorithm = merge_sort_paten
                sorting_algo_name = "Merge Sort"
            elif event.key == pygame.K_q and not sorting:
                sorting_algorithm = quick_sort_patern
                sorting_algo_name = "Quick Sort"
    pygame.quit()
if __name__ =="__main__":
    main()
