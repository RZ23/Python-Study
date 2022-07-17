from collections import deque
from collections import defaultdict
from collections import deque
import heapq
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
def print_matrix(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if isinstance(dp[i][j],int):
                print(format(dp[i][j],'02d'), end="|")
            else:
                print(dp[i][j], end="|")
        print()
def display_graph(node):
    if node is None:
        return []
    display_order = {}
    q= deque()
    q.append(node)
    visit = set()
    visit.add(node)
    result = []
    while len(q)>0:
        cur_node =q.popleft()
        print("{}".format(cur_node.val),end = ":")
        sub_result = []
        for neighbor in cur_node.neighbors:
            if neighbor not in visit:
                q.append(neighbor)
                visit.add(neighbor)
            print(neighbor.val,end = ",")
            sub_result.append(neighbor.val)
        print()
        # result.append(sub_result)
        display_order[cur_node.val] = sub_result
    for i in range(1,len(display_order.keys())+1):
        result.append(display_order[i])
    return result
def generate_graph(graph_node_list):
    if len(graph_node_list)==0:
        return None
    graph = []
    graph_map = {}
    for i in range(len(graph_node_list)):
        index = str(i+1)
        graph.append(Node(i+1))
        graph_map[index] = graph[i]
    for i in range(len(graph)):
        for adj in graph_node_list[i]:
            graph[i].neighbors.append(graph_map[str(adj)])
    return graph[0]
print("---------------------207. Course Schedule-------------------------")
def canFinish(numCourses,prerequisites):
    course_adj = {i:[] for i in range(numCourses)}
    for course,precouse in prerequisites:
        course_adj[course].append(precouse)
    visited = set()
    def dfs(course):
        if course in visited:
            return False
        if course_adj[course]==[]:
            return True
        visited.add(course)
        for precourse in course_adj[course]:
            if not dfs(precourse):
                return False
        visited.remove(course)
        course_adj[course]=[]
        return True
    for course in range(numCourses):
        if not dfs(course):
            return False
    return True
test_case = [[2,[[1,0]]],[2,[[1,0],[0,1]]],[5,[[0,1],[0,2],[1,3],[1,4],[3,4]]]]
for numsCourse,prerequisties in test_case:
    print(f"For the {numsCourse} courses, could finish all the course(s) "
          f"followed by the prerequisites rule {prerequisties}:{canFinish(numsCourse,prerequisties)}")
print("---------------------210. Course Schedule II-------------------------")
def findOrder(numCourses,prerequisites):
    prereq = {c:[] for c in range(numCourses)}
    for course,precourse in prerequisites:
        prereq[course].append(precourse)
    '''
    a course has three possible states:
    visited: course has been added to the output
    visiting: course not added to output, but add to determine the cycle
    unvisited: course not add to output or cyclse 
    '''
    output = []
    visited = set()
    visiting = set()
    def dfs(course):
        # if the course is in the cycle check
        if course in visiting:
            return False
        # if the couse is added to output
        if course in visited:
            return True
        # add to cycle check set
        visiting.add(course)
        # else deep first search all the precouse for it
        for precourses in prereq[course]:
            if not dfs(precourses):
                return False
        # this course will no longer in the path, so remove it
        # from the cycle
        visiting.remove(course)
        visited.add(course)
        output.append(course)
        return True
    for course in range(numCourses):
        if not dfs(course):
            return []
    return output
test_case =[[2,[[1,0]]],[4,[[1,0],[2,0],[3,1],[3,2]]],[1,[]],[2,[[1,0],[0,1]]]]
for numCourses, prerequisites in test_case:
    print(f"The order of {numCourses} course(s) with prerequisites {prerequisites} is {findOrder(numCourses,prerequisites)}")
print("---------------------200. Number of Islands-------------------------")
def numIslands(grid):
    row = len(grid)
    col = len(grid[0])
    def dfs(r,c):
        if (r<0 or c<0 or r>=row or c>=col or grid[r][c]!="1"):
            return
        grid[r][c]="x"
        dfs(r,c+1)
        dfs(r,c-1)
        dfs(r+1,c)
        dfs(r-1,c)
    if len(grid)==0 or len(grid[0])==0:
        return 0
    count = 0
    for r in range(row):
        for c in range(col):
            if grid[r][c]=="1":
                count = count+1
                dfs(r,c)
    print("The visited map is ")
    print_matrix(grid)
    return count
test_case =[[
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
],
[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]]
for grid in test_case:
    print(f"The Map is ")
    print_matrix(grid)
    print(f"There is(are) {numIslands(grid)} island(s) in this map")
print("---------------------133. Clone Graph-------------------------")
def cloneGraph(node):
    map_node = {}
    def dfs(node):
        if node in map_node:
            return map_node[node]
        new_node = Node(node.val)
        map_node[node] = new_node
        for neighbor in node.neighbors:
            new_node.neighbors.append(dfs(neighbor))
        return new_node
    if node is not None:
        return dfs(node)
    else:
        return None
test_case = [[[2,4],[1,3],[2,4],[1,3]],[[]],[]]
for node_list in test_case:
    node_root = generate_graph(node_list)
    print("The Original Graph is:")
    display_graph(node_root)
    print("The Colon Graph is:")
    coloned_node_root = cloneGraph(node_root)
    display_graph(coloned_node_root)
print("---------------------127. Word Ladder-------------------------")
'''
Using BFS algorithm
'''
def ladderLength(beginWord,endWord,wordList):
    if endWord not in wordList:
        return 0
    # using the collections.defaultdict set the default of dict to []
    neighbor = defaultdict(list)
    wordList.append(beginWord)
    # create the dict for each word pattern
    # like hot=> *ot,h*t,ho*
    for word in wordList:
        for j in range(len(word)):
            pattern = word[:j]+"*"+word[j+1:]
            neighbor[pattern].append(word)
    visited = set([beginWord])
    q = deque([beginWord])
    res = 1
    while len(q)>0:
        for i in range(len(q)):
            word=q.popleft()
            if word==endWord:
                return res
            for j in range(len(word)):
                pattern = word[:j]+"*"+word[j+1:]
                for neiword in neighbor[pattern]:
                    if neiword not in visited:
                        q.append(neiword)
        res = res+1
    return 0
test_case = [["hit","cog",["hot","dot","dog","lot","log","cog"]],["hit", "cog",["hot","dot","dog","lot","log"]]]
for beginWord,endWord,wordList in test_case:
    print(f"For the begin word {beginWord}, there is(are) {ladderLength(beginWord, endWord, wordList)} step(s)"
          f"to reach the end word {endWord} with dict {wordList}")
print("---------------------417. Pacific Atlantic Water Flow-------------------------")
'''
Using BFS algorithm
'''
def pacificAtlantic(heights):
    COL =len(heights[0])
    ROW = len(heights)
    Pac = set()
    Atl = set()
    def dfs(r,c,visit,prev):
        if r<0 or r==ROW or c<0 or c==COL or heights[r][c]<prev or (r,c) in visit:
            return
        visit.add((r,c))
        dfs(r+1,c,visit,heights[r][c])
        dfs(r-1,c,visit,heights[r][c])
        dfs(r, c+1, visit, heights[r][c])
        dfs(r, c-1, visit, heights[r][c])
    for r in range(ROW):
        dfs(r,0,Pac,heights[r][0])
        dfs(r,COL-1,Atl,heights[ROW-1][COL-1])
    for c in range(COL):
        dfs(0,c,Pac,heights[0][c])
        dfs(ROW-1,c,Atl,heights[ROW-1][c])
    result = []
    for r in range(ROW):
        for c in range(COL):
            if (r,c) in Pac and (r,c) in Atl:
                result.append([r,c])
    return result
test_case = [[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]],[[2,1],[1,2]]]
for heights in test_case:
    print("The Map is ")
    print_matrix(heights)
    print(f"The wave(s) could flow to both Pacific and Atlantic is(are) {pacificAtlantic(heights)}")
print("---------------------743. Network Delay Time-------------------------")
'''
Using BFS algorithm
'''
def networkDelayTime(times, n, k):
    edges = defaultdict(list)
    for start,end, weight in times:
        edges[start].append((end,weight))
    miniheap = [(0,k)]
    visited = set()
    t = 0
    while len(miniheap)>0:
        weight,start_node = heapq.heappop(miniheap)
        if start_node in visited:
            continue
        visited.add(start_node)
        t = max(t,weight)
        for end_node,visited_weight in edges[start_node]:
            if end_node not in visited:
                heapq.heappush(miniheap,((weight+visited_weight),end_node))
    if len(visited)==n:
        return t
    else:
        return -1

test_case = [[[[2,1,1],[2,3,1],[3,4,1]],4,2],[[[1,2,1]],2,1],[[[1,2,1]],2,2]]
for times, n,k in test_case:
    print(f"The Minimum edge from node {k} to reach all the {n} node(s) with {times} is {networkDelayTime(times, n, k)}")
print("---------------------79. Word Search-------------------------")
def exist(board, word):
    ROW = len(board)
    COL=len(board[0])
    path = set()
    def dfs(r,c,i):
        if i==len(word):
            return True
        if r<0 or c<0 or r==ROW or c==COL or word[i]!=board[r][c] or (r,c) in path:
            return False
        path.add((r,c))
        res = dfs(r+1,c,i+1) or dfs(r-1,c,i+1) or dfs(r,c-1,i+1) or dfs(r,c+1,i+1)
        path.remove((r,c))
        return res
    for r in range(ROW):
        for c in range(COL):
            if dfs(r,c,0):
                return True
    return False
test_case = [[[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"ABCCED"],\
            [[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"SEE"],\
            [[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"ABCB"]]
for board, word in test_case:
    print("The Board is")
    print_matrix(board)
    print(f"The word {word} can be found in the board: {exist(board, word)}")
print("---------------------463. Island Perimeter-------------------------")
def islandPerimeter(grid):
    ROW = len(grid)
    COL =len(grid[0])
    Perimeter = 0
    visit = set()
    def dfs(r,c):
        if r<0 or c<0 or r==ROW or c==COL or grid[r][c]==0:
            return 1
        if (r,c) in visit:
            return 0
        visit.add((r,c))
        Perimeter = dfs(r+1,c)
        Perimeter = Perimeter+dfs(r-1,c)
        Perimeter = Perimeter+dfs(r,c+1)
        Perimeter = Perimeter+dfs(r,c-1)
        return Perimeter
    for r in range(ROW):
        for c in range(COL):
            if grid[r][c]==1:
                return dfs(r,c)
def sizeofisland(grid):
    ROW =len(grid)
    COL= len(grid[0])
    visit = set()
    def dfs(r,c):
        if r<0 or r==ROW or c<0 or c==COL or grid[r][c]==0:
            return 0
        grid[r][c]=0
        return dfs(r+1,c)+dfs(r-1,c)+dfs(r,c+1)+dfs(r,c-1)+1
    for r in range(ROW):
        for c in range(COL):
            if grid[r][c]==1:
                return dfs(r,c)
test_case = [[[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]],[[1]],[[1,0]]]
for grid in test_case:
    print("The Map is ")
    print_matrix(grid)
    print(f"the Perimeter of the island is {islandPerimeter(grid)},the size of island is {sizeofisland(grid)}\n")
print("---------------------695. Max Area of Island-------------------------")
def maxAreaOfIsland(grid):
    ROW =len(grid)
    COL= len(grid[0])
    visit = set()
    max_size = 0
    def dfs(r,c):
        if r<0 or r==ROW or c<0 or c==COL or grid[r][c]==0:
            return 0
        grid[r][c]=0
        return dfs(r+1,c)+dfs(r-1,c)+dfs(r,c+1)+dfs(r,c-1)+1
    for r in range(ROW):
        for c in range(COL):
            if grid[r][c]==1:
                max_size = max(max_size,dfs(r,c))
    return max_size
test_case = [[[0,0,1,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,1,1,0,0,0],
             [0,1,1,0,1,0,0,0,0,0,0,0,0],
             [0,1,0,0,1,1,0,0,1,0,1,0,0],
             [0,1,0,0,1,1,0,0,1,1,1,0,0],
             [0,0,0,0,0,0,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,0,1,1,1,0,0,0],
             [0,0,0,0,0,0,0,1,1,0,0,0,0]],[[0,0,0,0,0,0,0,0]]]
for grid in test_case:
    print("Map is")
    print_matrix(grid)
    print(f"The Max Area of island is {maxAreaOfIsland(grid)}")
print("---------------------1466. Reorder Routes to Make All Paths Lead to the City Zero-------------------------")
def minReorder(n,connections):
    edge = {(a,b) for a,b in connections}
    neighbors = {city:[] for city in range(n)}
    visited = set()
    change = 0
    for start, end in connections:
        neighbors[start].append(end)
        neighbors[end].append(start)
    def dfs(city):
        nonlocal edge,neighbors,change
        for neighbor in neighbors[city]:
            if neighbor in visited:
                continue
            if (neighbor,city) not in edge:
                change = change+1
            visited.add(neighbor)
            dfs(neighbor)
    visited.add(0)
    dfs(0)
    return change
test_case = [[6, [[0,1],[1,3],[2,3],[4,0],[4,5]]],[3,[[1,0],[2,0]]]]
for n,connections in test_case:
    print(f"For {n} city(s), based on the current road map {connections}, there is(are)"
    f" {minReorder(n, connections)} road(s) need to update to make sure all the city(s)"
    f"could connection to city Zero")
print("---------------------684. Redundant Connection-------------------------")
'''
using Union and Find Algorithm
'''
def findRedundantConnection(edges):
    # set the parent list
    par = [i for i in range(len(edges)+1)]
    # set the rank list:
    rank = [1] *(len(edges)+1)
    # find function, find the parent of each node
    def find(n):
        p = par[n]
        # run the loop, until the parent of the node is not itself
        while p!=par[p]:
            par[p] = par[par[p]]
            p = par[p]
        return p
    # union function, union two nodes into one subset, with the higher rank
    def union(n1,n2):
        p1 = find(n1)
        p2 = find(n2)
        # if two noeds already in the same subset, return False
        if p1==p2:
            return False
        # if p1's rank is higher than p2's rank
        # then merge p2 into the p1,p2's parent is p1
        # and update p1's rank to add p2's rank
        if rank[p1]>rank[p2]:
            par[p2]=p1
            rank[p1] = rank[p1]+rank[p2]
        # if p2's rank is higher than p1's rank
        # then merge p1 into the p2,p1's parent is p2
        # and update p2's rank to add p1's rank
        else:
            par[p1]=p2
            rank[p2] = rank[p2]+rank[p1]
        return True
    for n1,n2 in edges:
        if not union(n1,n2):
            return [n1,n2]
test_case = [[[1,2],[1,3],[2,3]],[[1,2],[2,3],[3,4],[1,4],[1,5]]]
for edges in test_case:
    print(f"To create a tree, should remove the edge {findRedundantConnection(edges)} from the edge list {edges}")