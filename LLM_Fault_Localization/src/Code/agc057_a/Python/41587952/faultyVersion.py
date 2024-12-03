#ABC057A Antichain of Integer Strings

'''
前哨戦。

いや、なんだこれ、難しすぎるだろ。
下の桁ほど不利っぽいな。L,R=1,100 のとき、A={3}を選んでしまうと3のつく整数全滅だし。
逆に上の桁は貪欲してよかったりする？

どう考えても数字の大きいものから貪欲に決定するかんじ。
R = 32451 として考えると
10000 - 32451 は貪欲に採用できそう。
ああでも19999 を使っているから9999が採用できないか。それはかわいそうだ。
･･･いや、いうて10000 - 19999 を全採用できるほうが嬉しいから、それでいいのか。

R = 12451 のときは？
10000 - 12451 は貪欲に採用できる。とりあえず2451以下は全滅。あと1245以下。
それ以外、すなわち 2452 - 9999 は自由に選べそう？

R = 10451 のときは？
10000 - 10451 は貪欲、あと1000 - 1045 は禁止。それ以上は貪欲可能。

R = 10000 のときは？
このコーナーケース置いてくれるの優しすぎるだろ。
1000 - 1000 が禁止。 1000 - 0000 も禁止。それ以外は貪欲可。

なんかひどい回答になりそう。提出すっか。
'''
for _ in range(int(input())):
    L,R=map(int,input().split())
    if len(str(L))==len(str(R)): print(R-L+1); continue
    X=len(str(R))
    R_NG=max(min(10**(X-1)-1,R-10**(X-1)),int(str(R)[:-1]),L-1)
    ans=int(str(R)[1:])+1+10**(X-1)-1-R_NG
    print(ans)
