def print_matrix(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if isinstance(dp[i][j],int):
                print(format(dp[i][j],'02d'), end="|")
            else:
                print(dp[i][j], end="|")
        print()
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