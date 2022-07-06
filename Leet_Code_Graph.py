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
test_case = [[2,[[1,0]]],[2,[[1,0],[0,1]]]]
for numsCourse,prerequisties in test_case:
    print(f"For the {numsCourse} courses, could finish all the course(s) "
          f"followed by the prerequisites rule {prerequisties}:{canFinish(numsCourse,prerequisties)}")