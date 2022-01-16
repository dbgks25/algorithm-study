n,m = map(int,input().split())
result = 0
for i in range(n):
  lst = list(map(int, input().split()))
  min_val = min(lst)
  result = max(result,min_val)
print(result)

# 1이 될 때까지 문제
n, k = map(int, input().split())
# r은 n과 k의 나머지로 이 값을 빼면 나누기 과정만 함.
# 그리고 빼는 값의 크기가 1번 과정의 개수이므로 n에서 빼줌
r = n%k
n -= r
# n을 k로 나눠주면서 몫을 n에 저장(마지막에 n이 1이 되고, s를 더할 때 그냥 더해도 됨)
s = 0
while n != 1:
  n //= k
  s += 1

result = r + s
print(result)