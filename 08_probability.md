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

#### python
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

### 부분집합의 수
- 원소의 갯수가 N개인 집합은 2^N개의 부분집합을 갖는다.
- 모든 집합은 공집합과 자기 자신인 집합을 부분집합으로 갖는다.
```
A = set(["O", "A", "B"])
empty_set = set([])

''A의 부분집합의 갯수''
print("A의 길이 : {}, A의 부분집합의 갯수 : {}".format(len(A), 2**len(A))) 

''공집합과 A 자신이 A의 부분집합인지 확인''
print(empty_set <= A, A <= A)

=====<print>=====
A의 길이 : 3, A의 부분집합의 갯수 : 8
(True, True)
```

### 합집합과 교집합의 분배법칙
- 합, 곱 분배법칙 처럼 교집합과 합집합도 분배법칙 성립한다.
  - <img src="https://latex.codecogs.com/gif.latex?%5Cfn_cm%20A%5Ccup%20%28B%5Ccap%20C%29%20%3D%20%28A%20%5Ccup%20B%29%5Ccap%28A%20%5Ccup%20C%29">
  - <img src="https://latex.codecogs.com/gif.latex?%5Cfn_cm%20A%20%5Ccap%20%28B%20%5Ccup%20C%29%20%3D%20%28A%20%5Ccap%20B%29%20%5Ccup%20%28A%20%5Ccap%20C%29">

#### python
- 집합 A, B, C 에 대해서 교집합과 합집합의 분배법칙 성립 확인해보기.
```
A = frozenset([1, 3, 5])
B = frozenset([1, 2, 3])
C = frozenset([2, 4, 6])

print("A∪(B∩C)")
print(A.union(B.intersection(C)))
print((A.union(B)).intersection(A.union(C)))
print((A | B) & (A | C))
print("\n")
print("B∩(A∪C)")
print(B.intersection(A.union(C)))
print((B.intersection(A)).union(B.intersection(C)))
print((B & A) | (B & C))

=====<print>=====

A∪(B∩C)
frozenset({1, 2, 3, 5})
frozenset({1, 2, 3, 5})
frozenset({1, 2, 3, 5})


B∩(A∪C)
frozenset({1, 2, 3})
frozenset({1, 2, 3})
frozenset({1, 2, 3})
```

# 확률의 수학적 정의와 의미

### 표본공간과 확률표본
- ``확률표본 probabilistic sample, random sample``, ``표본 sample`` : 풀고자 하는 확률적 문제에서 발생(realize)할 수 있는 하나의 현상, 혹은 선택(sampled)될 수 있는 경우
- ``표본공간 sample space`` : 가능한 모든 표본의 집합 set
  - 표본공간을 정의한다는 것은 어떤 표본이 가능하고 어떤 표본이 가능하지 않은지를 정의하는 것과 같다.
- 표본공간은 풀고자 하는 문제에 대한 지식이나 필요성에 따라서 달라지게 된다.
  - 과일가게에서 과일을 선택하는 문제에서 그 가게에서 어떤 과일을 파느냐에 따라서 표본공간은 달라지게 된다.
- 표본이 연속적인 숫자이면 표본공간의 원소는 무한개일 수 있다.

#### 확률표본과 표본공간의 예
- 동전 던지기 : 동전을 한 번 던졌을 때 가능한 확률표본과 표본공간
  - 확률표본(=표본) : 앞면 H, 뒷면 T
  - 표본공간 : sample_space = {H, T}
- 동전 두번 던지기 : 
  - 확률표본 : HH, HT, TH, TT
  - 표본공간 : sample_space = {HH, HT, TH, TT}
- 트럼프 카드 뽑기 : 트럼프 카드에서 한 장의 카드를 뽑을 때의 확률표본과 표본공간
  - 확률표본 : ♤, ♥, ♧, ◆
  - 표본공간 : sample_space = {♤, ♥, ♧, ◆}
- 약속날짜를 잡으려고 할때 31일 인가? 
  - 확률표본 : 31일이 포함된 달만 가능하다. 1/1~1/31, 3/1~3/31, ..., 12/1~12/31
  - 표본공간 : sample_space = {1/1, 1/2,...,1/31,3/1,3/2,...,3/31,...,12/1,12/2,....,12/31}
- 과일가게에서 과일을 선택하는 경우
  - 확률표본 : A 가게 사과, 바나나, B 가게 복숭아, 사과 
  - A_store = {"apple", "banana"}, B_store = {"peach", "apple"}
- 연속적인 숫자로 된 표본의 표본공간
  - 주식 거래 가격 : sample_space = {x : -30 <= x <= 30}, 표본공간의 원소는 무한개이다.
  - 9월 요일별 평균 기온 : temp_space = {x : 10 <= x <= 28}, 표본공간의 원소는 무한개이다.
  - 회전하는 원판에 던진 화살의 각도 : arrow_space = {x : 0 < x <= 360}, 표본공간의 원손느 무한개이다.
  - 일반성인의 체온 맞추기 : body_temp_space = R, 표본공간은 실수 전체이다.

### 사건 
- ``사건 event`` :  표본공간의 부분집합. 전체 표본공간 중에서 현재 관심을 가지고 있는 일부 표본의 집합.
- 동전 던지기 표본공간에서 가능한 부분집합
  - A = {}, B = {H}, C = {T}, D = {H, T}
  - 사건은 전체표본 공간에서 나올 수 있는 경우를 의미한다. 즉 B 사건은 동전을 던졌을 때 H가 나올 경우이고, D 사건은 동전을 던졌을 때 H 또는 T 가 나오는 경우이다. 

#### python
- 동전 던지기 문제의 부분집합을 하나의 집합으로 합할 수 있다.
  - ``frozenset([]) 자료형은 딕셔너리의 key 로 사용할 수 있다. set([]) 자료형은 key 로 사용할 수 없다.``
```
A = frozenset([])
B = frozenset(['H'])
C = frozenset(['T'])
D = frozenset(['H', 'T'])

print(set([A, B, C, D]))

=====<print>=====

{frozenset(), frozenset({'H'}), frozenset({'T'}), frozenset({'T', 'H'})}
```
- 카드뽑기 문제의 표본공간의 모든 사건을 구하고 frozenset으로 만들고, set으로 합하기
```
play_card = frozenset([1, 2, 3, 4])
subset_1 = frozenset([1])
subset_2 = frozenset([2])
subset_3 = frozenset([3])
subset_4 = frozenset([4])
subset_5 = frozenset([1, 2])
subset_6 = frozenset([1, 3])
subset_7 = frozenset([1, 4])
subset_8 = frozenset([2, 3])
subset_9 = frozenset([2, 4])
subset_10 = frozenset([3, 4])
subset_11 = frozenset([1, 2, 3])
subset_12 = frozenset([1, 2, 4])
subset_13 = frozenset([2, 3, 4])
subset_14 = frozenset([1, 3, 4])
subset_15 = frozenset([])
subset_16 = frozenset([1, 2, 3, 4])

ttl_set = set([subset_1, subset_2, subset_3, subset_4,\
subset_5, subset_6, subset_7, subset_8, subset_9,\
subset_10, subset_11, subset_12,subset_13, subset_14,\
subset_15, subset_16])
    
print(ttl_set)

=====<print>=====

{frozenset({3, 4}), frozenset({2}), frozenset({1, 4}), frozenset({2, 3, 4}), frozenset({2, 3}), frozenset({1, 2, 4}), frozenset({1, 2}), frozenset({3}), frozenset({2, 4}), frozenset({1}), frozenset(), frozenset({1, 2, 3, 4}), frozenset({1, 2, 3}), frozenset({1, 3}), frozenset({1, 3, 4}), frozenset({4})}
```
- 동전 2번 던지기의 표본공간과 가능한 모든 사건을 frozenset으로 만들고, set으로 합하기
```
coin_problem = {"HH", "HT", "TH", "TT"}
c1 = frozenset(["HH"])
c2 = frozenset(["HT"])
c3 = frozenset(["TH"])
c4 = frozenset(["TT"])
c5 = frozenset(["HH", "HT"])
c6 = frozenset(["HH", "TH"])
c7 = frozenset(["HH", "TT"])
c8 = frozenset(["HT", "TH"])
c9 = frozenset(["HT", "TT"])
c10 = frozenset(["TH", "TT"])
c11 = frozenset(["HH", "HT", "TH"])
c12 = frozenset(["HH", "HT", "TT"])
c13 = frozenset(["HH", "TH", "TT"])
c14 = frozenset(["HT", "TH", "TT"])
c15 = frozenset([])
c16 = frozenset(["HH", "HT", "TH", "TT"])

ttl_set = set([c1, c2, c3, c4, c5, c6, c7, c8,\
c9, c10, c11, c12, c13, c14, c15, c16])

print(ttl_set)

=====<print>=====

{frozenset({'HH', 'TT', 'TH'}), frozenset({'HH'}), frozenset({'HH', 'TH', 'HT'}), frozenset({'HH', 'HT'}), frozenset({'HH', 'TT', 'TH', 'HT'}), frozenset({'HH', 'TT'}), frozenset({'HH', 'TT', 'HT'}), frozenset({'TT', 'HT'}), frozenset({'TT'}), frozenset({'TT', 'TH', 'HT'}), frozenset({'TH'}), frozenset(), frozenset({'TH', 'HT'}), frozenset({'HH', 'TH'}), frozenset({'HT'}), frozenset({'TT', 'TH'})}
```

























