# 집합

### 집합과 원소
- ``집합 set`` : 구별가능하는 객체의 모임
- ``원소 element`` : 집합에 포함 된 구별 가능한 객체들
- 집합의 표기 : A = {1, 2, 3}, B = {H, T}
  - 집합을 이루는 객체가 반드시 숫자일 필요 없음
  - C = {★, ♤, ♥}

#### python
- ``set([])`` : 내용을 변경할 수 있다, 뮤터블 mutable 자료형
  - 뮤터블 자료형은 딕셔너리 자료형의 key 값이 될 수 없음 주의
- ``frozenset([])`` : 내용을 변경할 수 없다, 임뮤터블 immutable 자료형
```
A = set([1, 2, 3, 4])
B = frozenset(['X', 'Y', 'Z'])
C = {"\u2700", "\u2701"}

=====<print>=====

{1, 2, 3, 4}
frozenset({'Y', 'X', 'Z'})
{'✁', '✀'}
```

### 집합의 크기
- ``집합의 크기 cardinality`` : 집합이 가지는 원소의 갯수, |A| 로 표시하거나 card 기호를 사용하기도 한다.
  - A = {2, 4, 6, 8} 의 크기
  - |A| = card(A) = 4
- 두 실수 사이에서는 무한개의 원소를 가질 수 있다. 이러한 경우는 set, frozenset 으로 나타낼 수 없다.
  - D = {x: 0 < x <= 1} 

#### python
```
len_A = len(A)
len_B = len(B)
len_C = len(C)

=====<print>=====
4
3
2
```
### 합집합과 교집합
- ``합집합 union`` : 각 집합의 원소를 모두 포함하는 집합
  - 기호 : A∪B
- ``교집합 intersection`` : 두 사건 모두에 속하는 원소로만 이루어진 집합
  - 기호 : B∩C 

#### python
- 합집합 : A.union(B), A | B
- 교집합 : A.intersection(B), A & B
```
A1 = set([1, 2, 3, 4])
A2 = set([2, 4, 6])
A3 = set([1, 2, 3])
A4 = set([2, 3, 4, 5, 6])

print("합집합 {} union {} = {} ".format(A1, A2, A1.union(A2)))
print("합집합 {} \ {} = {}".format(A2, A1, A2 | A1))
print("교집합 {} intersection {} = {}".format(A3, A4, A3.intersection(A4)))
print("교집합 {} & {} = {}".format(A3, A4, A3 & A4)

=====<print>=====

합집합 {1, 2, 3, 4} union {2, 4, 6} = {1, 2, 3, 4, 6} 
합집합 {2, 4, 6} \ {1, 2, 3, 4} = {1, 2, 3, 4, 6}
교집합 {1, 2, 3} intersection {2, 3, 4, 5, 6} = {2, 3}
교집합 {1, 2, 3} & {2, 3, 4, 5, 6} = {2, 3}
```

### 전체집합, 부분집합, 여집합
- 부분집합 subset : 어떤 집합의 원소 중 일부만을 포함하는 집합
- 전체집합 : 부분집합의 원래집합
  - A ⊂ Ω (A는 Ω의 부분집합이다.)
  - ```A ⊂ A, for all A (모든 집합은 자기 자신의 부분집합이다.)```
- 진부분집합 proper subset : 원소의 크기가 원래 집합보다 더 작은 부분집합
  - Q = {1,2,3}, A={1,2} 이면 A는 Q의 부분집합이면서 진부분집합이다.
  - Q는 Q의 부분집합이지만 진부분집합은 아니다.

#### python
- 부분집합 : A.issubset(B), A <= B, B >= A (True, False 반환)
- 진부분집합 : A < B, B > A (True, False 반환)
```
A1 = set([1, 2, 3, 4])
A2 = set([2, 4, 6])
A3 = set([1, 2, 3])
A4 = set([2, 3, 4, 5, 6])

print("부분집합 {} issubset {} = {}".format(A2, A4, A2.issubset(A4)))
print("부분집합 {} <= {} = {}".format(A2, A4, A2 <= A4))
print("부분집합 {} <= {} = {}".format(A4, A2, A4 <= A2))
print("부분집합 {} >= {} = {}".format(A2, A2, A2 >= A2))
print("진부분집합 {} < {} = {}".format(A3, A1, A3 < A1))
print("진부분집합 {} > {} = {}".format(A1, A3, A1 > A3))
print("진부분집합 {} > {} = {}".format(A1, A1, A1 > A1))

=====<print>=====

분집합 {2, 4, 6} issubset {2, 3, 4, 5, 6} = True
부분집합 {2, 4, 6} <= {2, 3, 4, 5, 6} = True
부분집합 {2, 3, 4, 5, 6} <= {2, 4, 6} = False
부분집합 {2, 4, 6} >= {2, 4, 6} = True
진부분집합 {1, 2, 3} < {1, 2, 3, 4} = True
진부분집합 {1, 2, 3, 4} > {1, 2, 3} = True
진부분집합 {1, 2, 3, 4} > {1, 2, 3, 4} = False
```

### 차집합과 여집합
- ``차집합 difference`` : A에 속하면서 B에는 속하지 않는 원소로 이루어진 A의 부분집합
  - A - B (A에서 B를 뺸 집합)
- ``여집합 complement`` : 전체집합 Ω 중에서 부분집합 A에 속하지 않는 원소로만 이루어진 집합
  - A^c
- 여집합 A^c는 전체집합에서 집합 A를 뺀 차집합과 같다.

#### python
- 차집합 : A.difference(B), A - B
```
A1 = set([1, 2, 3, 4])
A2 = set([2, 4, 6])
A3 = set([1, 2, 3])
A4 = set([2, 3, 4, 5, 6])

print("차집합 {} difference {} = {}".format(A1, A2, A1.difference(A2)))
print("차집합 {}-{} = {}".format(A3, A4, A3 - A4))

=====<print>=====

차집합 {1, 2, 3, 4} difference {2, 4, 6} = {1, 3}
차집합 {1, 2, 3}-{2, 3, 4, 5, 6} = {1}
```

### 공집합
- 공집합 null set : 아무런 원소도 포함하지 않는 집합
  - 기호 : ∅
- 공집합은 모든 집합의 부분집합이다.
  -  ∅ ⊂ A, for all A
- 공집합과 임의의 집합의 교집합은 공집합이다.
  -  ∅ ∩ A = ∅
- 공집합과 임의의 집합의 합집합은 그 집합 자신이다.
  -  ∅ ∪ A = A
- 여집합과 원래의 집합의 교집합은 공집합이다.
  - A ∩ A^c = ∅ 

### python
- 공집합 정의
```
empty_set = set([])
print(empty_set)

=====<print>=====

set()
```
- 집합 A와 공집합의 관계
- 공집합은 임의의 집합의 부분집합이면서 진부분집합이다.
```
A1 = set([1, 2, 3, 4])
empty_set = set([])

print("empty_set < {} = {}".format(A1, empty_set < A1))
print("empty_set <= {} = {}".format(A1, empty_set <= A1))
print("empty_set intersection {} = {}".format(A1, empty_set.intersection(A1)))
print("empty_set union {} = {}".format(A1, empty_set.union(A1)))

=====<print>=====

empty_set < {1, 2, 3, 4} = True
empty_set <= {1, 2, 3, 4} = True
empty_set intersection {1, 2, 3, 4} = set()
empty_set union {1, 2, 3, 4} = {1, 2, 3, 4}
```
