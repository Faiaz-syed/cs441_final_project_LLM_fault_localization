n = int(input())
s = input()
ans = -1
cnt = 0
for i in range(n):
    if s[i] == "o":
        cnt += 1
    if s[i] == "-" or i == n-1:
        if s != "o"*n and cnt != 0:
            ans = max(ans,cnt)
            cnt = 0
print(ans)