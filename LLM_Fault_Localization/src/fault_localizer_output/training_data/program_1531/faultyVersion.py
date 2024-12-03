import math
from bisect import bisect_left, bisect_right
from typing import Generic, Iterable, Iterator, TypeVar, Union, List

T = TypeVar('T')

# https://github.com/tatyam-prime/SortedSet/blob/main/SortedSet.py
class SortedSet(Generic[T]):
	BUCKET_RATIO = 50
	REBUILD_RATIO = 170
	
	def _build(self, a=None) -> None:
		"Evenly divide `a` into buckets."
		if a is None: a = list(self)
		size = self.size = len(a)
		bucket_size = int(math.ceil(math.sqrt(size/self.BUCKET_RATIO)))
		self.a = [a[size*i//bucket_size: size*(i+1)//bucket_size] for i in range(bucket_size)]
	
	def __init__(self, a: Iterable[T] = []) -> None:
		"Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
		a = list(a)
		if not all(a[i] < a[i+1] for i in range(len(a)-1)):
			a = sorted(set(a))
		self._build(a)
	
	def __iter__(self) -> Iterator[T]:
		for i in self.a:
			for j in i: yield j
	
	def __reversed__(self) -> Iterator[T]:
		for i in reversed(self.a):
			for j in reversed(i): yield j
	
	def __len__(self) -> int:
		return self.size
	
	def __repr__(self) -> str:
		return "SortedSet"+str(self.a)
	
	def __str__(self) -> str:
		s = str(list(self))
		return "{"+s[1: len(s)-1]+"}"
	
	def _find_bucket(self, x: T) -> List[T]:
		"Find the bucket which should contain x. self must not be empty."
		for a in self.a:
			if x <= a[-1]: return a
		return a
	
	def __contains__(self, x: T) -> bool:
		if self.size == 0: return False
		a = self._find_bucket(x)
		i = bisect_left(a, x)
		return i != len(a) and a[i] == x
	
	def add(self, x: T) -> bool:
		"Add an element and return True if added. / O(√N)"
		if self.size == 0:
			self.a = [[x]]
			self.size = 1
			return True
		a = self._find_bucket(x)
		i = bisect_left(a, x)
		if i != len(a) and a[i] == x: return False
		a.insert(i, x)
		self.size += 1
		if len(a) > len(self.a)*self.REBUILD_RATIO:
			self._build()
		return True
	
	def discard(self, x: T) -> bool:
		"Remove an element and return True if removed. / O(√N)"
		if self.size == 0: return False
		a = self._find_bucket(x)
		i = bisect_left(a, x)
		if i == len(a) or a[i] != x: return False
		a.pop(i)
		self.size -= 1
		if len(a) == 0: self._build()
		return True
	
	def lt(self, x: T) -> Union[T, None]:
		"Find the largest element < x, or None if it doesn't exist."
		for a in reversed(self.a):
			if a[0] < x:
				return a[bisect_left(a, x)-1]
	
	def le(self, x: T) -> Union[T, None]:
		"Find the largest element <= x, or None if it doesn't exist."
		for a in reversed(self.a):
			if a[0] <= x:
				return a[bisect_right(a, x)-1]
	
	def gt(self, x: T) -> Union[T, None]:
		"Find the smallest element > x, or None if it doesn't exist."
		for a in self.a:
			if a[-1] > x:
				return a[bisect_right(a, x)]
	
	def ge(self, x: T) -> Union[T, None]:
		"Find the smallest element >= x, or None if it doesn't exist."
		for a in self.a:
			if a[-1] >= x:
				return a[bisect_left(a, x)]
	
	def __getitem__(self, x: int) -> T:
		"Return the x-th element, or IndexError if it doesn't exist."
		if x < 0: x += self.size
		if x < 0: raise IndexError
		for a in self.a:
			if x < len(a): return a[x]
			x -= len(a)
		raise IndexError
	
	def index(self, x: T) -> int:
		"Count the number of elements < x."
		ans = 0
		for a in self.a:
			if a[-1] >= x:
				return ans+bisect_left(a, x)
			ans += len(a)
		return ans
	
	def index_right(self, x: T) -> int:
		"Count the number of elements <= x."
		ans = 0
		for a in self.a:
			if a[-1] > x:
				return ans+bisect_right(a, x)
			ans += len(a)
		return ans

from collections import defaultdict

v = SortedSet()
v.add((10**9+5<<30)+0)
ar = defaultdict(int)
n = int(input())
mask = (1 << 30) - 1
for i in range(n):
	l, r = map(int,input().split())
	r += 1
	tt = v.lt(l<<30) # l 未満
	te = v.ge(l<<30) # l 以上

	if tt == None:
		while te != None:
			teval = te >> 30
			teind = te & mask
			if teval == 10**9+5:
				v.discard(te)
				v.add((10**9+5<<30)+r-l)
				break
			tr = te
			te = v.gt(te)
			tenind = te & mask
			if r-l < tenind:
				v.discard(tr)
				v.add((teval+r-l-teind<<30)+r-l)
				break
			v.discard(tr)
		v.add((l<<30)+0)
		continue
	
	ttval = tt>>30
	ttind = tt&mask
	teval = te>>30
	teind = te&mask

	if ttval + teind - ttind <= l:
		gogo = teind
	else:
		gogo = ttind + l - ttval
	
	while te != None:
		teval = te >> 30
		teind = te & mask
		if teval == 10**9+5:
			v.discard(te)
			v.add((10**9+5<<30)+gogo+r-l)
			break
		tr = te
		te = v.gt(te)
		tenind = te & mask
		if tenind > gogo+r-l:
			v.discard(tr)
			v.add((teval+gogo+r-l-teind<<30)+gogo+r-l)
			break
		v.discard(tr)
	v.add((l<<30)+gogo)

print(v.ge((10**9+2)<<30)&mask)