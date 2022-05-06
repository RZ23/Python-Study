def print_matrix(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if isinstance(dp[i][j],int):
                print(format(dp[i][j],'02d'), end="|")
            else:
                print(dp[i][j], end="|")
        print()
print("---------------------362. Unique Paths-------------------------")

def uniquePaths(m,n):
    matrix = [[1 for _ in range(n)] for _ in range(m)]
    for row in range(1,m):
        for col in range(1,n):
            matrix[row][col] =matrix[row][col-1]+matrix[row-1][col]
    print_matrix(matrix)
    return matrix[m-1][n-1]
test_case = [[3,7],[3,2],[1,5],[5,1]]
for m,n in test_case:
    print("There are {} ways to get from left-top to right-bottom in the {} * {} grid".format(uniquePaths(m,n),m,n))