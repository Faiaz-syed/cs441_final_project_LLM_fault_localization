L,R,LL,RR = map(int,input().split())
l = max(L,LL)
r = min(R,RR)
ans = max(r-l+1,0)
print(ans)