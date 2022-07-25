from collections import deque
from collections import defaultdict
from collections import deque
import heapq
import collections
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
print("---------------------178 · Graph Valid Tree-------------------------")
'''
O(E+V)
'''
def valid_tree(n,edges):
    if not n:
        return False
    # create the adj list
    adj = {i:[] for i in range(n)}
    # update the adj list
    for start,end in edges:
        adj[start].append(end)
        adj[end].append(start)
    # set the visit set
    visit = set()
    # create the dfs function, add the prev variable to store the
    # parent node of this node
    def dfs(i,prev):
        if i in visit:
            return False
        visit.add(i)
        for j in adj[i]:
            if j==prev:
                continue
            if not dfs(j,i):
                return False
        return True
    # set the prev for the first node to 0
    return dfs(0,-1) and n==len(visit)
test_case=[[5,[[0, 1], [0, 2], [0, 3], [1, 4]]],[5 ,[[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]]]
for n, edges in test_case:
    print(f"based on the edges {edges}, this {n} node(s) could create a tree: {valid_tree(n, edges)}")
print("---------------------212. Word Search II-------------------------")
print("***** Method One: Add Prun Word Function *****")
class TrieNode():
    def __init__(self):
        self.children = {}
        self.isWord = False
    def addWord(self,word):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur=cur.children[c]
        cur.isWord = True
    def prunWord(self,word):
        cur= self
        nodeAndChildKey = []
        for char in word:
            nodeAndChildKey.append((cur,char))
            cur = cur.children[char]
        for parentNode,childKey in reversed(nodeAndChildKey):
            targetNode = parentNode.children[childKey]
            if len(targetNode.children)==0:
                del parentNode.children[childKey]
            else:
                return
def findWords(board,words):
    root = TrieNode()
    for w in words:
        root.addWord(w)
    ROW = len(board)
    COL = len(board[0])
    result = []
    visit = set()
    def dfs(r,c,node,word):
        if (r<0 or r==ROW or c<0 or c==COL or board[r][c] not in node.children or (r,c) in visit):
            return
        visit.add((r,c))
        node = node.children[board[r][c]]
        word = word+board[r][c]
        if node.isWord:
            result.append(word)
            node.isWord = False
            root.prunWord(word)
        dfs(r+1,c,node,word)
        dfs(r-1,c,node,word)
        dfs(r,c+1,node,word)
        dfs(r,c-1,node,word)
        visit.remove((r,c))
    for r in range(ROW):
        for c in range(COL):
            dfs(r,c,root,"")
    return result
test_case = [[[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],["oath","pea","eat","rain"]],
[[["a","b"],["c","d"]],["abcb"]]]
for board,words in test_case:
    print("The board is")
    print_matrix(board)
    print(f"based on the board and given word list {words}")
    print(f"Could find word(s) {findWords(board, words)} in the board")
print("***** Method Two: Without Prun Word Function")
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False
    def addWord(self,word):
        cur = self
        for char in word:
            if char not in cur.children:
                cur.children[char] = TrieNode()
            cur = cur.children[char]
        cur.isWord = True
def findWords(board,words):
    root = TrieNode()
    for w in words:
        root.addWord(w)
    ROW = len(board)
    COL =len(board[0])
    result = set()
    visit = set()
    def dfs(r,c,node,word):
        if (r<0 or c<0 or r==ROW or c==COL or board[r][c] not in node.children or (r,c) in visit):
            return
        visit.add((r,c))
        node = node.children[board[r][c]]
        word = word+board[r][c]
        if node.isWord:
            result.add(word)
        dfs(r-1,c,node,word)
        dfs(r+1,c,node,word)
        dfs(r,c+1,node,word)
        dfs(r,c-1,node,word)
        visit.remove((r,c))
    for r in range(ROW):
        for c in range(COL):
            dfs(r,c,root,"")
    return list(result)
test_case = [[[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],["oath","pea","eat","rain"]],
[[["a","b"],["c","d"]],["abcb"]]]
for board,words in test_case:
    print("The board is")
    print_matrix(board)
    print(f"based on the board and given word list {words}")
    print(f"Could find word(s) {findWords(board, words)} in the board")
print("---------------------323. Number of Connected Components in an Undirected Graph-------------------------")
print("***** Method One: Find and Union Algorithm *****")
def countComponents(n,edges):
    par = [i for i in range(n+1)]
    rank = [1]*n
    def find_parent(n):
        result = n
        while result!=n:
            par[n] = par[par[n]]
            n = par[n]
        return n
    def union(n1,n2):
        p1,p2 = find_parent(n1),find_parent(n2)
        if p1==p2:
            return 0
        if rank[p2]>rank[p1]:
            par[p1] = p2
            rank[p2]=rank[p2]+rank[p1]
        else:
            par[p2]=p1
            rank[p1] = rank[p1]+rank[p2]
        return 1
    result = n
    for n1,n2 in edges:
        result = result - union(n1,n2)
    return result
test_case = [[5,[[0, 1], [1, 2], [3, 4]]],[5,[[0, 1], [1, 2], [2, 3], [3, 4]]],[4,[[2,3]]],
             [7,[[0,1],[0,4],[2,3],[2,5],[2,6],[0,2]]]]
for n,edges in test_case:
    print(f"Based on the edge(s) {edges}, there is(are) {countComponents(n,edges)} component(s) in this {n} node(s) graph")
print("***** Method Two: DFS algorithm *****")
def countComponents_dfs(n,edges):
    adj = {i:[] for i in range(n)}
    for start, end in edges:
        adj[start].append(end)
        adj[end].append(start)
    visit = set()
    def dfs(node):
        if node in visit:
            return 0
        visit.add(node)
        for neighbor in adj[node]:
            dfs(neighbor)
        return 1
    result = 0
    for i in range(n):
        result = result+dfs(i)
    return result
for n,edges in test_case:
    print(f"Based on the edge(s) {edges}, there is(are) {countComponents_dfs(n,edges)} component(s) in this {n} node(s) graph")
print("---------------------1584. Min Cost to Connect All Points-------------------------")
'''
Using Prim's algorithm, with Visit Set and MiniHeap
'''
def minCostConnectPoints(points):
    N = len(points)
    # create the adj list
    adj = {i:[] for i in range(N)}
    #fill the adj list
    for i in range(N):
        x1,y1 = points[i]
        for j in range(i+1,N):
            x2,y2 = points[j]
            distance = abs(x1-x2)+abs(y1-y2)
            adj[i].append([distance,j])
            adj[j].append([distance,i])
    # Prim Algorithm
    result = 0
    visit = set()
    minHeap = [[0,0]] # [cost, end_point]
    while (len(visit))<N:
        cost,i = heapq.heappop(minHeap)
        if i in visit:
            continue
        result = result+cost
        visit.add(i)
        for cost,neigh in adj[i]:
            if neigh not in visit:
                heapq.heappush(minHeap,[cost,neigh])
    return result
test_case = [[[0,0],[2,2],[3,10],[5,2],[7,0]],[[3,12],[-2,5],[-4,1]]]
for points in test_case:
    print(f"Based on the point(s), to connect all the point(s) {points}, "
          f"the shortest distance is {minCostConnectPoints(points)} ")
print("---------------------269. Alien Dictionary-------------------------")
'''
using postorder dfs and topologic
'''
def alienOeder(words):
    # set the hashmap for each character
    adj = {ch: set() for word in words for ch in word }
    for i in range(len(words)-1):
        w1,w2 = words[i],words[i+1]
        min_len = min(len(w1),len(w2))
        # if w1 = abc, w2 = ab, since the words are sorted
        # in this case, return false
        if len(w1)>len(w2) and w1[:min_len]==w2[:min_len]:
            return ""
        #fill the hashmap, by compare each character
        # find the first different character, and
        # append this character to first character's adj list
        for j in range(min_len):
            if w1[j]!=w2[j]:
                adj[w1[j]].add(w2[j])
                break
        # after finish the hashmap, use the DFS to find the cycle
        # if there is a cycle existed, reture " "
        visit = {} # False=visted, True = current path
        res = []
    def dfs(c):
        '''
        if character is visited, return the visit[c]'s value
        Ture:means character is on the current path, False: means character is just visited
        example: ab, and abc, using postorder dfs, a->b and a->c, after mark c as visited,
        when process the b->c, and found c in the visited, not means there is the cycle
        '''
        if c in visit:
            return visit[c]
        visit[c] = True
        for neighbor in adj[c]:
            if dfs(neighbor):
                return True
        visit[c] = False
        res.append(c)
    # get the result list
    for c in adj:
        if dfs(c):
            return ""
    res.reverse()
    return "".join(res)
test_case = [["caa", "aaa", "aab"],["wrt","wrf","er","ett","rftt"]]
for words in test_case:
    print("Based on the {}, the order is {}".format(words,alienOeder(words)))
print("---------------------1905. Count Sub Islands-------------------------")
def countSubIslands(grid1,grid2):
    ROW = len(grid2)
    COL =len(grid2[0])
    visit = set()
    def dfs(r,c):
        if (r<0 or c<0 or r==ROW or c==COL or (r,c) in visit or grid2[r][c]==0):
            return True
        visit.add((r,c))
        result = True
        if grid1[r][c]==0:
            result = False
        result = dfs(r+1,c) and result
        result = dfs(r-1,c) and result
        result = dfs(r,c+1) and result
        result = dfs(r,c-1) and result
        return result
    count = 0
    for r in range(ROW):
        for c in range(COL):
            if grid2[r][c]==1 and (r,c) not in visit and dfs(r,c):
                count = count+1
    return count
test_case =[[[[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]],
            [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]],\
           [[[1,0,1,0,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,0,1,0,1]],
            [[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[1,0,0,0,1]]]]
for grid1,grid2 in test_case:
    print("Map One:")
    print_matrix(grid1)
    print("Map Two:")
    print_matrix(grid2)
    print(f"For map Two, There is(are) {countSubIslands(grid1, grid2)} sub-islands in map One")
print("---------------------778. Swim in Rising Water-------------------------")
def swimInWater(grid):
    N = len(grid)
    visit =set()
    mheap = [[grid[0][0],0,0]] # (time/max-height,row,col)
    direction = [[0,1],[0,-1],[1,0],[-1,0]]
    visit.add((0,0))
    while mheap:
        t,r,c = heapq.heappop(mheap)
        if r==N-1 and c==N-1:
            return t
        for dr,dc in direction:
            neiR,neiC = r+dr,c+dc
            if (neiR<0 or neiC<0 or neiC==N or neiR==N or (neiR,neiC) in visit):
                continue
            visit.add((neiR,neiC))
            heapq.heappush(mheap,[max(t,grid[neiR][neiC]),neiR,neiC])
test_case = [[[0,2],[1,3]],[[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]]
for grid in test_case:
    print("Map is:")
    print_matrix(grid)
    print(f"Based on the map,The least time can reach the bottom right is {swimInWater(grid)}  ")
    print()
print("---------------------286.Walls and Gates-------------------------")
def walls_and_gates(rooms):
    ROW = len(rooms)
    COL = len(rooms[0])
    visit = set()
    q = deque()
    def addRoom(r,c):
        if (r<0 or r==ROW or c<0 or c==COL or (r,c) in visit or rooms[r][c]==-1):
            return
        visit.add((r,c))
        q.append([r,c])
    # find the gate,add to queue
    for r in range(ROW):
        for c in range(COL):
            if rooms[r][c]==0:
                q.append([r,c])
                visit.add((r,c))
    dist = 0
    # using BFS and pop from the queue
    while q:
        # pop the items by level
        for i in range(len(q)):
            r,c = q.popleft()
            rooms[r][c]=dist
            addRoom(r+1,c)
            addRoom(r-1,c)
            addRoom(r,c+1)
            addRoom(r,c-1)
        dist = dist+1


def print_door_and_well_map(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if dp[i][j]==2147483647:
                print("E",end = "|")
            elif dp[i][j]==-1:
                print("W",end = "|")
            elif dp[i][j]==0:
                print("G", end="|")
            else:
                print(dp[i][j],end = "|")
        print()
test_case = [[[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]],
[[0,-1],[2147483647,2147483647]]]
for i,rooms in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print("The Map is")
    print_door_and_well_map(rooms)
    print("The distance from each room to nearest gate is")
    walls_and_gates(rooms)
    print_door_and_well_map(rooms)
print("---------------------130. Surrounded Regions-------------------------")
def solve(board):
    ROW = len(board)
    COL = len(board[0])
    def capture(r,c):
        if (r<0 or c<0 or r==ROW or c==COL or board[r][c]!="O"):
            return
        board[r][c]="T"
        capture(r+1,c)
        capture(r-1,c)
        capture(r,c+1)
        capture(r,c-1)
    # capture unsurrounded regions (O -> T)
    for r in range(ROW):
        for c in range(COL):
            if (board[r][c]=="O" and (r in [0,ROW-1] or c in [0,COL-1])):
                capture(r,c)
    # capture surronded regions (O->X)
    for r in range(ROW):
        for c in range(COL):
            if board[r][c]=="O":
                board[r][c]="X"
    # Uncapture unsurrounded regions (T->O)
    for r in range(ROW):
        for c in range(COL):
            if board[r][c]=="T":
                board[r][c]="O"
test_case = [[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]],[["X"]]]
for i, board in enumerate(test_case):
    print(f"##### Test Case {i+1} #####")
    print("The original Map is")
    print_matrix(board)
    solve(board)
    print("The Captured Map is")
    print_matrix(board)
print("---------------------787. Cheapest Flights Within K Stops-------------------------")
print("***** Method One: Bellman-Ford Algorithm *****")
def findCheapestPrice(n,flights,src,dst,k):
    prices = [float("inf")]*n
    prices[src]=0
    for i in range(k+1):
        tempPrice = prices.copy()
        for s,d,p in flights:
            if prices[s]==float('inf'):
                continue
            if prices[s]+p<tempPrice[d]:
                tempPrice[d] = prices[s]+p
        prices = tempPrice
    return -1 if prices[dst]==float("inf") else prices[dst]
test_case = [[4,[[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]],0,3,1],\
            [3,[[0,1,100],[1,2,100],[0,2,500]],0,2,1],
            [3,[[0,1,100],[1,2,100],[0,2,500]],0,2,0]]
for n,flights,src,dst,k in test_case:
    print(f"Based on the flight fare table {flights}, the cheapest fare from "
              f"srouce {src} to destination {dst} with almost {k} stop(s) is {findCheapestPrice(n, flights, src, dst, k)}")
print("***** Method Two: DFS Algorithm *****")
def findCheapestPrice_dfs(n,flights,src,dst,k):
    def dfs(stop,numOfStop):
        if numOfStop>k:
            return float("inf")
        elif stop ==dst:
            return 0
        cheapestFlight = float("inf")
        for flight in flights:
            if flight[0]==stop:
                cheapestFlight = min(cheapestFlight,(flight[2]+dfs(flight[1],numOfStop+1)))
        return cheapestFlight
    cheapestFlight = dfs(src,-1)
    return cheapestFlight if cheapestFlight!=float("inf") else -1
for n,flights,src,dst,k in test_case:
    print(f"Based on the flight fare table {flights}, the cheapest fare from "
              f"srouce {src} to destination {dst} with almost {k} stop(s) is {findCheapestPrice_dfs(n, flights, src, dst, k)}")
print("***** Method Three: BFS Algorithm *****")
def findCheapestPrice_bfs(n,flights,src,dst,k):
    graph = collections.defaultdict(dict)
    for u,v,w in flights:
        graph[u][v]=w
    q = deque()
    q.append([src,0])
    ans,step = float("inf"),0
    while q:
        for _ in range(len(q)):
            cur,cost = q.popleft()
            if cur==dst:
                ans = min(ans,cost)
                continue
            for v,w in graph[cur].items():
                if cost+w>ans:
                    continue
                q.append((v,cost+w))
        if step>k:
            break
        step = step+1
    return -1 if ans == float("inf") else ans
for n,flights,src,dst,k in test_case:
    print(f"Based on the flight fare table {flights}, the cheapest fare from "
              f"srouce {src} to destination {dst} with almost {k} stop(s) is {findCheapestPrice_bfs(n, flights, src, dst, k)}")