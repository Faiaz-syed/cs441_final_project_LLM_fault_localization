#!/usr/bin/env python3
# from typing import *
from math import sqrt


# def solve(N: int) -> int:
def solve(N):
    # 片方を決め打てばよさそう？
    ans = 10**20
    if N == 0:
        return 0
    for n in range(600000):
        m =  (-20 *  n**3 + 3 * sqrt(3) * sqrt(16 * n**6 - 40 * n**3 * N + 27 * N**2) + 27 *  N)**(1/3)/(3 * 2**(1/3)) - (2 * 2**(1/3) * n**2)/(3 * (-20 * n**3 + 3 * sqrt(3) * sqrt(16*  n**6 - 40* n**3 *N + 27 *N**2) + 27* N)**(1/3)) - n/3
        for m1 in range(int(m)+10,int(m)-10,-1):
            if m1 < 0:
                break

            if n**3+n*m1**2+n**2 *m1 + m1**3 >= N:
                ans = min(ans, n**3+n*m1**2+n**2 *m1 + m1**3)
    return ans
# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    a = solve(N)
    print(a)

if __name__ == '__main__':
    main()