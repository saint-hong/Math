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
```python
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
```python
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
```python
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
```python
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
```python
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
```python
empty_set = set([])
print(empty_set)

=====<print>=====

set()
```
- 집합 A와 공집합의 관계
- 공집합은 임의의 집합의 부분집합이면서 진부분집합이다.
```python
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
```python
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
```python
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
- ``사건 event`` :  표본공간의 **부분집합**. 전체 표본공간 중에서 현재 관심을 가지고 있는 일부 표본의 집합.
  - 사건=부분집합
- 동전 던지기 표본공간에서 가능한 부분집합
  - A = {}, B = {H}, C = {T}, D = {H, T}
  - 사건은 전체표본 공간에서 나올 수 있는 경우를 의미한다. 즉 B 사건은 동전을 던졌을 때 H가 나올 경우이고, D 사건은 동전을 던졌을 때 H 또는 T 가 나오는 경우이다. 

#### python
- 동전 던지기 문제의 부분집합을 하나의 집합으로 합할 수 있다.
  - ``frozenset([]) 자료형은 딕셔너리의 key 로 사용할 수 있다. set([]) 자료형은 key 로 사용할 수 없다.``
```python
A = frozenset([])
B = frozenset(['H'])
C = frozenset(['T'])
D = frozenset(['H', 'T'])

print(set([A, B, C, D]))

=====<print>=====

{frozenset(), frozenset({'H'}), frozenset({'T'}), frozenset({'T', 'H'})}
```
- 카드뽑기 문제의 표본공간의 모든 사건을 구하고 frozenset으로 만들고, set으로 합하기
```python
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
```python
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

- **itertools의 chain과 combinations를 사용**
- 여러개의 부분집합을 하나의 set([])에 넣기위해서 반복문을 사용하여 앞의 것을 넣고 새로 저장한 후 뒤의 것과 합하는 방법을 생각해 보았으나 실패
- 대신 기본 데이터를 조합해주는 기능인 combinations 를 사용하면 간편해진다.

```python
from itertools import chain, combinations

omega = {"HH", "HT", "TH", "TT"}


def get_make_subset(x) :
    return set([frozenset(c) for c in chain.from_itertools(combinations(omega, n) for n in rnage(len(x)+1))])


get_make_subset(omega)

=====<print>=====

{frozenset(),
 frozenset({'HT', 'TT'}),
 frozenset({'TH', 'TT'}),
 frozenset({'HT', 'TH'}),
 frozenset({'TT'}),
 frozenset({'HT', 'TH', 'TT'}),
 frozenset({'HH'}),
 frozenset({'HH', 'TT'}),
 frozenset({'HH', 'HT'}),
 frozenset({'TH'}),
 frozenset({'HH', 'TH'}),
 frozenset({'HT'}),
 frozenset({'HH', 'HT', 'TH'}),
 frozenset({'HH', 'TH', 'TT'}),
 frozenset({'HH', 'HT', 'TT'}),
 frozenset({'HH', 'HT', 'TH', 'TT'})}
```

# 확률
- ``확률 probability`` : 사건(부분집합)을 입력하면 숫자가 출력되는 **함수**
  - 확률은 함수와 같다. 사건과 사건이 발생할 확률을 관계지어 준다.
- 확률의 정의역 : 표본공간의 모든 사건(부분집합)의 집합
  - 확률의 범위는 엄격하게는 시그마대수(sigma algebra)라는 특별한 사건의 집합에 대해서만 정의된다.
  - 참고 링크 
    - *확률의 수학적 정의* https://arca.live/b/maths/1137243;
    - *확률 개념의 발전과정* https://pkjung.tistory.com/167;
- 확률은 모든 각각의 사건(부분집합)에 어떤 **숫자를 할당(assign, allocate)하는 함수**이다.
  - P(A) : A라는 사건(부분집합)에 할당된 숫자
  - P({H}) 는 "H라는 표본이 선택될 확률", P({H, T}) 는 "H 또는 T라는 표본이 선택될 확률"이라는 의미
- 확c률이라는 함수를 정의한다는 것은 "A가 선택될 확률이 얼마인가?"라는 질문에 대한 답을 모든 경우(사건, 부분집합)에 대해서 **미리 준비해 놓은 것** 또는 **할당해 놓은 것**과 같다.

### 콜모고로프의 공리
- ``콜모고로프의 공리 kolmogorov's axioms`` : 확률이라는 함수를 정의하는 3가지 규칙
- 모든 확률은 콜모고로프의 공리를 따라야 한다. 
  1. 모든 사건에 대해 확률은 실수이고 0 또는 양수이다.
      - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%29%20%5Cleq%200">
  2. 표본공간(전체집합)이라는 사건(부분집합)에 대한 확률은 1이다.
      - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28%5COmega%29%20%3D%201">
  3. 공통 원소가 없는 두 사건의 합집합의 확률은 사건별 확률의 합이다.
      - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20A%20%5Ccap%20B%20%3D%20%5Cvarnothing%20%5Crightarrow%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29">
      - 공통원소가 없는 두 사건, 즉 교집합이 공집합인 두 집합을 **서로소 pairwise disjoint set** 라고 한다.
      - 참고 링크 "서로소와 파티션" https://juggernaut.tistory.com/entry/%EC%84%9C%EB%A1%9C%EC%86%8CPairwise-Disjoint-Set-%EC%99%80-%ED%8C%8C%ED%8B%B0%EC%85%98Partition;


### 확률은 표본이 아닌 사건을 입력으로 가지는 함수
- 확률은 표본을 입력받는 것이 아닌 사건(부분집합)을 입력받는 함수이다.
- 주사위를 던졌을 때 숫자 1이 나올 확률
  - 틀린 표기 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%281%29%20%3D%20%5Cdfrac%7B1%7D%7B6%7D">
  - 맞는 표기 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28%5Cleft%20%5C%7B%201%20%5Cright%20%5C%7D%29%20%3D%20%5Cdfrac%7B1%7D%7B6%7D">
- 표본 즉 확률표본은 표본공간의 구성 재료이다. 사건은 표본공간에서 우리가 관심을 갖는 부분집합이다.

#### python
- 확률은 딕셔너리 타입을 사용하여 나타낼 수 있다. 집합을 정의할 때 frozenset 으로 해야 딕셔너리의 key로 사용할 수 있다.
- 사건 : key, 확률 : value
- 동전을 한 번 던질때를 확률로 나타내면,
```python
A = frozenset([])
B = frozenset(["H"])
C = frozenset(["T"])
D = frozenset(["H", "T"])

P = {A:0, B:0.4, C:0.6, D:1}
print(P)

=====<print>=====

{frozenset(): 0, frozenset({'H'}): 0.4, frozenset({'T'}): 0.6, frozenset({'T', 'H'}): 1}
```

### 확률이 콜모고로프의 공리를 만족하는지 확인
- 확률은 사건에 대해 확률값을 할당하는 함수이고 콜모고로프의 공리를 만족해야 한다.
- 즉 콜모고로프의 공리만 만족한다면 사건에 대해 할당하는 확률값은 정해진 것이 아니다.
- 동전 던지기 문제에서 앞면이 나올 확률이 반드시 1/2 이 아니어도 된다.
  - P({∅}) = 0, P({H}) = 0.3, P({T}) = 0.7, P({H, T}) = 1
  - P({∅}) = 0, P({H}) = 0.82, P({T}) = 0.18, P({H, T}) = 1
  - **P({∅}) = 0, P({H}) = 0.55, P({T}) = 0.45, P({H, T}) = 1**
- 이와 같이 확률값을 할당했다면 콜모고로프의 공리를 만족하는지 확인해야 한다. 
  1. 모든 확률은 0 이상이다.
      - P({∅}) >= 0 , P({H}) >= 0, P({T}) >= 0, P({H, T}) >= 0
  2. 전체집합에 대한 확률은 1이다.
      - P(Ω) = P({H, T}) = 1
  3. 교집합이 공집합인 사건(부분집합)의 합집합인 사건의 확률은 각 사건(부분집합)의 확률의 합이다. 
      - P({H}) = 0.55 = P({H}∪∅) = 0.55 + 0
      - P({T}) = 0.45 = P({T}∪∅) = 0.45 + 0
      - P({H, T}) = 1 = P({H, T}∪∅) = 1 + 0
      - P({H, T}) = 1 = P({H}∪{T}) = 0.55 + 0.45

### ``확률은 골동품 가게의 포장된 상품(사건, 부분집합)과 같다.``
- 표본이 아닌 사건에 대해 확률값을 할당하는 확률의 의미는 다음과 같다.
  - 가게에서 파는 하나 하나의 골동품은 표본이다. 모든 골동품은 서로 다르기때문에 똑같은 물건(표본)은 없다.
  - 가게에서 파는 모든 골동품의 집합은 표본공간(전체집합)이다.
  - 사건(부분집합)은 골동품을 넣은 상자를 말한다. 상자안의 골동품 개수에는 제한이 없다.
  - 상자안의 골동품이 하나가 될 수도 있고, 골동품이 없는 빈 포장(공집합)도 가능하다.
  - 확률은 상자에 붙인 가격표의 숫자이다. 가격은 마음데로 붙여도 되지만, 다음 규칙을 지켜야한다.
    1. 음수인 가격은 없다. 공짜(0)나 양수이어야 한다.
    2. 가게안의 모든 골동품을 하나의 상자에 포장하면 그 상자의 가격은 1이다.
    3. 공통적으로 포함된 골동품이 없는 두개의 상자의 가격은 그 두개의 포장에 들어간 골동품을 합쳐서하나의 상자로 만들었을 때의 가격과 같아야 한다. 즉 상자를 나누거나 합쳤다고 가격이 달라져셔는 안된다.

#### python
- 플레잉카드 한 장을 뽑아서 무늬를 결정하는 문제에 대해서 확률값을 할당하고 파이썬으로 구현해보기
```python

''부분집합 정의 (16 개)''
card_sub_1 = frozenset([])
card_sub_2 = frozenset(["S"])
card_sub_3 = frozenset(["H"])
card_sub_4 = frozenset(["C"])
card_sub_5 = frozenset(["D"])
card_sub_6 = frozenset(["S", "H"])
card_sub_7 = frozenset(["S", "C"])
card_sub_8 = frozenset(["S", "D"])
card_sub_9 = frozenset(["H", "C"])
card_sub_10 = frozenset(["H", "D"])
card_sub_11 = frozenset(["C", "D"])
card_sub_12 = frozenset(["S", "H", "C"])
card_sub_13 = frozenset(["S", "H", "D"])
card_sub_14 = frozenset(["H", "C", "D"])
card_sub_15 = frozenset(["S", "C", "D"])
card_sub_16 = frozenset(["H", "S", "C", "D"])

''확률값 정의''
S = 0.25
H = 0.38
C = 0.17
D = 0.20

card_proba = {
    card_sub_1 : 0,
    card_sub_2 : 0.25,
    card_sub_3 : 0.38,
    card_sub_4 : 0.17, 
    card_sub_5 : 0.20,
    card_sub_6 : S + H,
    card_sub_7 : S + C,
    card_sub_8 : S + D,
    card_sub_9 : H + C,
    card_sub_10 : H + D,
    card_sub_11 : C + D,
    card_sub_12 : S + H + C,
    card_sub_13 : S + H + D,
    card_sub_14 : H + C + D,
    card_sub_15 : S + C + D,
    card_sub_16 : H + S + C + D,
}

''콜모고로프의 공리 검증''
print("=== 콜모고로프의 공리 ===")
for i in range(1, 17) :
    name = eval('card_sub_'+str(i))
    print("P({}) = {}".format(name, round(card_proba[name], 2)), ",", "P({}) >= 0 --> {}".format(name, card_proba[name]>=0))

=====<print>=====

=== 콜모고로프의 공리  ===
P(frozenset()) = 0 , P(frozenset()) >= 0 --> True
P(frozenset({'S'})) = 0.25 , P(frozenset({'S'})) >= 0 --> True
P(frozenset({'H'})) = 0.38 , P(frozenset({'H'})) >= 0 --> True
P(frozenset({'C'})) = 0.17 , P(frozenset({'C'})) >= 0 --> True
P(frozenset({'D'})) = 0.2 , P(frozenset({'D'})) >= 0 --> True
P(frozenset({'H', 'S'})) = 0.63 , P(frozenset({'H', 'S'})) >= 0 --> True
P(frozenset({'C', 'S'})) = 0.42 , P(frozenset({'C', 'S'})) >= 0 --> True
P(frozenset({'D', 'S'})) = 0.45 , P(frozenset({'D', 'S'})) >= 0 --> True
P(frozenset({'C', 'H'})) = 0.55 , P(frozenset({'C', 'H'})) >= 0 --> True
P(frozenset({'D', 'H'})) = 0.58 , P(frozenset({'D', 'H'})) >= 0 --> True
P(frozenset({'D', 'C'})) = 0.37 , P(frozenset({'D', 'C'})) >= 0 --> True
P(frozenset({'C', 'H', 'S'})) = 0.8 , P(frozenset({'C', 'H', 'S'})) >= 0 --> True
P(frozenset({'D', 'H', 'S'})) = 0.83 , P(frozenset({'D', 'H', 'S'})) >= 0 --> True
P(frozenset({'D', 'C', 'H'})) = 0.75 , P(frozenset({'D', 'C', 'H'})) >= 0 --> True
P(frozenset({'D', 'C', 'S'})) = 0.62 , P(frozenset({'D', 'C', 'S'})) >= 0 --> True
P(frozenset({'D', 'C', 'S', 'H'})) = 1.0 , P(frozenset({'D', 'C', 'S', 'H'})) >= 0 --> True
```
### 주사위의 확률을 모든 사건(부분집합)에 대해 할당하기.
- P({1}) = 0.5, P({6}) = 0
- 표본공간 정의
```pyhon
cube_space = {1, 2, 3, 4, 5, 6}
cube_space

=====<print>=====

{1, 2, 3, 4, 5, 6}
```

- 모든 사건(부분집합) 정의
```python
cube1 = frozenset([1])
cube2 = frozenset([2])
cube3 = frozenset([3])
cube4 = frozenset([4])
cube5 = frozenset([5])
cube6 = frozenset([6])
cube7 = frozenset([1, 2])
cube8 = frozenset([1, 3])
cube9 = frozenset([1, 4])
cube10 = frozenset([1, 5])
cube11 = frozenset([1, 6])
cube12 = frozenset([2, 3])
cube13 = frozenset([2, 4])
cube14 = frozenset([2, 5])
cube15 = frozenset([2, 6])
cube16 = frozenset([3, 4])
cube17 = frozenset([3, 5])
cube18 = frozenset([3, 6])
cube19 = frozenset([4, 5])
cube20 = frozenset([4, 6])
cube21 = frozenset([5, 6])
cube22 = frozenset([1, 2, 3])
cube23 = frozenset([1, 2, 4])
cube24 = frozenset([1, 2, 5])
cube25 = frozenset([1, 2, 6])
cube26 = frozenset([1, 3, 4])
cube27 = frozenset([1, 3, 5])
cube28 = frozenset([1, 3, 6])
cube29 = frozenset([1, 4, 5])
cube30 = frozenset([1, 4, 6])
cube31 = frozenset([1, 5, 6])
cube32 = frozenset([2, 3, 4])
cube33 = frozenset([2, 3, 5])
cube34 = frozenset([2, 3, 6])
cube35 = frozenset([2, 4, 5])
cube36 = frozenset([2, 4, 6])
cube37 = frozenset([2, 5, 6])
cube38 = frozenset([3, 4, 5])
cube39 = frozenset([3, 4, 6])
cube40 = frozenset([3, 5, 6])
cube41 = frozenset([4, 5, 6])
cube42 = frozenset([1, 2, 3, 4])
cube43 = frozenset([1, 2, 3, 5])
cube44 = frozenset([1, 2, 3, 6])
cube45 = frozenset([1, 2, 4, 5])
cube46 = frozenset([1, 2, 4, 6])
cube47 = frozenset([1, 2, 5, 6])
cube48 = frozenset([1, 3, 4, 5])
cube49 = frozenset([1, 3, 4, 6])
cube50 = frozenset([1, 3, 5, 6])
cube51 = frozenset([1, 4, 5, 6])
cube52 = frozenset([2, 3, 4, 5])
cube53 = frozenset([2, 3, 4, 6])
cube54 = frozenset([2, 3, 5, 6])
cube55 = frozenset([2, 4, 5, 6])
cube56 = frozenset([3, 4, 5, 6])
cube57 = frozenset([1, 2, 3, 4, 5])
cube58 = frozenset([1, 2, 3, 4, 6])
cube59 = frozenset([1, 2, 3, 5, 6])
cube60 = frozenset([1, 2, 4, 5, 6])
cube61 = frozenset([1, 3, 4, 5, 6])
cube62 = frozenset([2, 3, 4, 5, 6])
cube63 = frozenset([1, 2, 3, 4, 5, 6])
cube64 = frozenset([])
```

- 각 부분집합에 확률 할당
```python
c1 = 0.5
c2 = 0.12
c3 = 0.145
c4 = 0.13
c5 = 0.105
c6 = 0

c1 + c2 + c3 + c4 + c5 + c6

=====<print>=====

1.0

cube_proba = {
    cube1 : c1,
    cube2 : c2,
    cube3 : c3,
    cube4 : c4,
    cube5 : c5,
    cube6 : c6,
    cube7 : c1 + c2,
    cube8 : c1 + c3,
    cube9 : c1 + c4,
    cube10 : c1 + c5,
    cube11 : c1 + c6,
    cube12 : c2 + c3,
    cube13 : c2 + c4,
    cube14 : c2 + c5,
    cube15 : c2 + c6,
    cube16 : c3 + c4,
    cube17 : c3 + c5,
    cube18 : c3 + c6,
    cube19 : c4 + c5,
    cube20 : c4 + c6,
    cube21 : c5 + c6,
    cube22 : c1 + c2 + c3,
    cube23 : c1 + c2 + c4,
    cube24 : c1 + c2 + c5,
    cube25 : c1 + c2 + c6,
    cube26 : c1 + c3 + c4,
    cube27 : c1 + c3 + c5,
    cube28 : c1 + c3 + c6,
    cube29 : c1 + c4 + c5,
    cube30 : c1 + c4 + c6,
    cube31 : c1 + c5 + c6,
    cube32 : c2 + c3 + c4,
    cube32 : c2 + c3 + c4,
    cube33 : c2 + c3 + c5,
    cube34 : c2 + c3 + c6,
    cube35 : c2 + c4 + c5,
    cube36 : c2 + c4 + c6,
    cube37 : c2 + c5 + c6,
    cube38 : c3 + c4 + c5,
    cube39 : c3 + c4 + c6,
    cube40 : c3 + c5 + c6,
    cube41 : c4 + c5 + c6,
    cube42 : c1 + c2 + c3 + c4,
    cube43 : c1 + c2 + c3 + c5,
    cube44 : c1 + c2 + c3 + c6,
    cube45 : c1 + c2 + c4 + c5,
    cube46 : c1 + c2 + c4 + c6,
    cube47 : c1 + c2 + c5 + c6,
    cube48 : c1 + c3 + c4 + c5,
    cube49 : c1 + c3 + c4 + c6,
    cube50 : c1 + c3 + c5 + c6,
    cube51 : c1 + c4 + c5 + c6,
    cube52 : c2 + c3 + c4 + c5,
    cube53 : c2 + c3 + c4 + c6,
    cube54 : c2 + c3 + c5 + c6,
    cube55 : c2 + c4 + c5 + c6,
    cube56 : c3 + c4 + c5 + c6,
    cube57 : c1 + c2 + c3 + c4 + c5,
    cube58 : c1 + c2 + c3 + c4 + c6,
    cube59 : c1 + c2 + c3 + c5 + c6,
    cube60 : c1 + c2 + c4 + c5 + c6,
    cube61 : c1 + c3 + c4 + c5 + c6,
    cube62 : c2 + c3 + c4 + c5 + c6,
    cube63 : c1 + c2 + c3 + c4 + c5 + c6,
    cube64 : 0
}

cube_proba

=====<print>=====

{frozenset({1}): 0.5,
 frozenset({2}): 0.12,
 frozenset({3}): 0.145,
 frozenset({4}): 0.13,
 frozenset({5}): 0.105,
 frozenset({6}): 0,
 frozenset({1, 2}): 0.62,
 frozenset({1, 3}): 0.645,
 frozenset({1, 4}): 0.63,
 frozenset({1, 5}): 0.605,
 frozenset({1, 6}): 0.5,
 frozenset({2, 3}): 0.265,
 frozenset({2, 4}): 0.25,
 frozenset({2, 5}): 0.22499999999999998,
 frozenset({2, 6}): 0.12,
 frozenset({3, 4}): 0.275,
 frozenset({3, 5}): 0.25,
 frozenset({3, 6}): 0.145,
 frozenset({4, 5}): 0.235,
 frozenset({4, 6}): 0.13,
 frozenset({5, 6}): 0.105,
 frozenset({1, 2, 3}): 0.765,
 frozenset({1, 2, 4}): 0.75,
 frozenset({1, 2, 5}): 0.725,
 frozenset({1, 2, 6}): 0.62,
 frozenset({1, 3, 4}): 0.775,
 frozenset({1, 3, 5}): 0.75,
 frozenset({1, 3, 6}): 0.645,
 frozenset({1, 4, 5}): 0.735,
 frozenset({1, 4, 6}): 0.63,
 frozenset({1, 5, 6}): 0.605,
 frozenset({2, 3, 4}): 0.395,
 frozenset({2, 3, 5}): 0.37,
 frozenset({2, 3, 6}): 0.265,
 frozenset({2, 4, 5}): 0.355,
 frozenset({2, 4, 6}): 0.25,
 frozenset({2, 5, 6}): 0.22499999999999998,
 frozenset({3, 4, 5}): 0.38,
 frozenset({3, 4, 6}): 0.275,
 frozenset({3, 5, 6}): 0.25,
 frozenset({4, 5, 6}): 0.235,
 frozenset({1, 2, 3, 4}): 0.895,
 frozenset({1, 2, 3, 5}): 0.87,
 frozenset({1, 2, 3, 6}): 0.765,
 frozenset({1, 2, 4, 5}): 0.855,
 frozenset({1, 2, 4, 6}): 0.75,
 frozenset({1, 2, 5, 6}): 0.725,
 frozenset({1, 3, 4, 5}): 0.88,
 frozenset({1, 3, 4, 6}): 0.775,
 frozenset({1, 3, 5, 6}): 0.75,
 frozenset({1, 4, 5, 6}): 0.735,
 frozenset({2, 3, 4, 5}): 0.5,
 frozenset({2, 3, 4, 6}): 0.395,
 frozenset({2, 3, 5, 6}): 0.37,
 frozenset({2, 4, 5, 6}): 0.355,
 frozenset({3, 4, 5, 6}): 0.38,
 frozenset({1, 2, 3, 4, 5}): 1.0,
 frozenset({1, 2, 3, 4, 6}): 0.895,
 frozenset({1, 2, 3, 5, 6}): 0.87,
 frozenset({1, 2, 4, 5, 6}): 0.855,
 frozenset({1, 3, 4, 5, 6}): 0.88,
 frozenset({2, 3, 4, 5, 6}): 0.5,
 frozenset({1, 2, 3, 4, 5, 6}): 1.0,
 frozenset(): 0}
```

- set 만들기 함수와 if 문을 사용하여 간편하게 만들 수 있다.
- set 에서 부분집합을 하나씩 꺼내고 부분집합안에 특정 표본이 있으면 이에 맞는 확률값을 더해주는 방식
```python

def get_set_of_subset(x) :

    from itertools import chain, combinations

    return set([frozenset(c) for c in chain.from_iterable(combinations(x, n) for n in range(len(x)+1))])

omega = {1, 2, 3, 4, 5, 6}
SS3 = get_set_of_subset(omega)

P3 = {}

for i in SS3 :
    probability = 0.0
    if 1 in i :
        probability += 0.5
    if 2 in i :
        probability += 0.12
    if 3 in i :
        probability += 0.145
    if 4 in i :
        probability += 0.13
    if 5 in i :
        probability += 0.105
    if 6 in i :
        probability += 0

    P3[i] = probability

P3

=====<print>=====

{frozenset({1, 3, 5}): 0.75,
 frozenset({1, 4}): 0.63,
 frozenset({4, 6}): 0.13,
 frozenset({2, 3}): 0.265,
 frozenset({2, 3, 4}): 0.395,
 frozenset({1, 4, 5, 6}): 0.735,
 frozenset({2, 6}): 0.12,
 frozenset({4, 5}): 0.235,
 frozenset({2, 3, 5, 6}): 0.37,
 frozenset({1, 2, 3, 6}): 0.765,
 frozenset({1}): 0.5,
 frozenset({1, 2, 3, 4, 5, 6}): 1.0,
 frozenset({2, 4, 5, 6}): 0.355,
 frozenset({4, 5, 6}): 0.235,
 frozenset({1, 3, 4, 6}): 0.775,
 frozenset({2, 3, 4, 6}): 0.395,
 frozenset({1, 2, 3, 4, 6}): 0.895,
 frozenset({1, 3, 4, 5, 6}): 0.88,
 frozenset({1, 4, 6}): 0.63,
 frozenset({3, 4}): 0.275,
 frozenset({2, 4, 6}): 0.25,
 frozenset({3, 4, 5, 6}): 0.38,
 frozenset({1, 2, 3, 4, 5}): 1.0,
 frozenset({2, 4}): 0.25,
 frozenset({5, 6}): 0.105,
 frozenset({2, 3, 5}): 0.37,
 frozenset({3, 4, 6}): 0.275,
 frozenset({3, 5}): 0.25,
 frozenset({2, 5, 6}): 0.22499999999999998,
 frozenset({3, 4, 5}): 0.38,
 frozenset({1, 6}): 0.5,
 frozenset({1, 2, 3}): 0.765,
 frozenset({1, 2, 4, 6}): 0.75,
 frozenset({1, 3}): 0.645,
 frozenset({1, 2, 6}): 0.62,
 frozenset({1, 3, 4}): 0.775,
 frozenset({1, 2, 4, 5}): 0.855,
 frozenset({1, 3, 5, 6}): 0.75,
 frozenset({1, 2, 3, 5, 6}): 0.87,
 frozenset({3, 6}): 0.145,
 frozenset({2, 3, 6}): 0.265,
 frozenset({2, 3, 4, 5}): 0.5,
 frozenset({1, 2, 5}): 0.725,
 frozenset({1, 5}): 0.605,
 frozenset({1, 4, 5}): 0.735,
 frozenset({1, 2, 3, 4}): 0.895,
 frozenset({2, 3, 4, 5, 6}): 0.5,
 frozenset({5}): 0.105,
 frozenset({4}): 0.13,
 frozenset({2}): 0.12,
 frozenset({1, 2, 4, 5, 6}): 0.855,
 frozenset({1, 2, 4}): 0.75,
 frozenset({1, 2}): 0.62,
 frozenset({1, 3, 6}): 0.645,
 frozenset({1, 2, 5, 6}): 0.725,
 frozenset({1, 2, 3, 5}): 0.87,
 frozenset({1, 3, 4, 5}): 0.88,
 frozenset({2, 4, 5}): 0.355,
 frozenset({2, 5}): 0.22499999999999998,
 frozenset({3, 5, 6}): 0.25,
 frozenset({3}): 0.145,
 frozenset({6}): 0.0,
 frozenset(): 0.0,
 frozenset({1, 5, 6}): 0.605}
```

### 확률값의 할당은 고정 된 것이 아니다.
- 주사위를 던졌을 때 어떤 숫자가 나올 확률은 왜 1/6 이라고 할까?
- 일반적으로 주사위가 **공정한 fair 주사위이며 공정하지 않다고 생각할 수 있는 증거가 없다는 가정을 전제하고 있기 때문이다.**
- 주사위의 특정한 면을 더 잘나오도록 주사위의 어떤 면을 깎거나, 주사위의 질량을 특정 부분에 더 치우치게 만든다면, 이 주사위의 확률은 1/6이 아니게 된다.
- 즉 주사위의 확률이 1/6 이라는 것은 확률값을 할당하는 하나의 방법일 뿐이며, 현실에서 반드시 이와 같이 확률값을 할당할 이유는 없다.
- 따라서 표본의 개수가 유한하고 각 사건에 대해 원소의 개수 이외의 아무런 정보가 없다면 각 사건의 확률을 다음과 같이 생각 할 수 있다.
  - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%29%20%3D%20%5Cdfrac%7Bcard%20%28A%29%7D%7Bcard%20%28%5COmega%29%7D">
  - 사건 A의 갯수 / 표본공간의 갯수
- 만약 새로운 정보 (자연법칙, 도메인 지식)나 표본에 대한 새로운 데이터가 존재하면 **더 믿을 수 있는 확률값**을 가정할 수 있다.

# 확률의 의미
- ``빈도주의적 관점 Frequentist`` : 반복적으로 선택된 표본이 사건(부분집합) A의 원소가 될 경향(propensity) 
  - 앞면이 나올 확률값이 0.5 라는 것은 동전을 반복하여 던졌을 경우 동전을 던진 전체 횟수에 확률값을 곱한 숫자만큼 해당 사건이 발생함, 이러한 반복적 경향을 갖는다고 봄.
  - P(H) = 0.5 이면, 동전 던지기 10,000번 X 0.5 = 5,000번 은 앞면이 나온다.
- ``베이지안 관점 Bayesian`` : 선택된 표본이 특정한 사건(부분집합)에 속한다는 가설(hypothesis), 명제(proposition) 혹은 주장(assertion)의 신뢰도(degree of belief)라고 본다. 베이지안 관점에서 반복이라는 개념은 사용되지 않는다.
  - 베이지안 관점에서는 확률값 :  사건이 발생할 가능성에 대한 신뢰도
  - 베이지안 관점에서 사건(부분집합)이란 : 원하는 답(표본)이 포함되어 있을 가능성이 있는 후보의 집합"과 같다. 예를들면 사건 {"HH"} 는 "HH"라는 표본이 들어있는 집합 중 하나이다.
  - 이러한 맥락에서 어떤 사건(부분집합)을 제시하면 그 자체로 "이 사건(부분집합)에 속한 원소 중에 우리가 원하는 답(표본)이 있다"는 명제 혹은 주장을 제시하는 것과 같다.
  - 또한 베이지안 관점에서 어떤 사건이 일어났다(occur) 또는 발생했다(realized)는 말은 그 사건(부분집합)의 원소 중에 정말로 선택된 표본, 우리가 구하고자 하는 표본이 있다는 사실을 알게 되었다는 것을 의미한다. 다른 의미로는 해당 사건이 말하고 있는 주장이 진실임을 알게 되었다는 뜻으로 지금까지 모르고 있던 추가적인 정보가 들어왔다는 것을 의미한다.
- 빈도주의적 관점과 베이지안 관점은 혼합되서 사용되기도 한다.
  - 암 환자일 확률 90%라는 검진 결과는 의사의 입장에서는 빈도주의적 관점이지만, 환자의 입장에서는 그 주장의 신뢰도라는 베이지안 관점으로 보게 된다. 

# 확률의 성질

- 표본공간, 표본, 사건 정의
- 어떤 집합 사람 20명 : 표본공간 sample space 의 원소의 개수는 20
  - 여자 / 남자, 머리카락 긴사람 / 짧은 사람 : 사람 한명은 표본 하나
- 0명이상의 사람을 선택했을 때 그 사람의 집합을 사건이라고 부른다.
  - 남자만 모여있는 부분집합 A = 사건
  - 머리카락 긴 사람이 모여있는 부분집합 B = 사건
  - 생일 i월 인 사람별로 부분집합 C1 - C12 = 사건
  - 원소 개수가 0인 부분집합도 사건이다. : 2월 생일인 사람이 없으면 C2의 원소개수는 0이다.
- 각 사건에 확률값 할당해보기 
  - 사건에 속하는 표본수에 비례하는 확률값 : A 사건이 10명이면 확률값은 10/20 = 1/2
  - 남자만 있는 부분집합 A : 12명, 확률값 : 2/6
  - 머리가 긴 사람의 부분집합 B : 9명, 확률값 : 1/6
  - 월별 생일자 부분집합 C1-C2 : 

### 성질 1. 공집합의 확률
- 공집합인 사건(부분집합)의 확률은 0이다.
- <img src="https://latex.codecogs.com/gif.latex?P%28%5Cvarnothing%29%20%3D%200"> 
  
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29%2C%20%5C%3B%5C%3B%20if%20%5C%3B%5C%3B%20P%28B%29%20%3D%20%5Cvarnothing%2C%5C%3B%20A%20%5Ccup%20%5Cvarnothing%20%3D%20A%20%5Crightarrow"> \
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5C%3B%5C%3B%20P%28A%20%5Ccup%20%5Cvarnothing%29%20%3D%20P%28A%29%20%3D%20P%28A%29%20&plus;%20P%28%5Cvarnothing%29"> \
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Ctherefore%20P%28%5Cvarnothing%29%20%3D%200">

### 성질2.. 여집합의 확률
- 어떤 사건의 여집합인 사건의 확률은 (1-원래 사건의 확률)과 같다.
- <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%5Ec%29%20%3D%201%20-%20P%28A%29">

  - A : 남자 부분집합, A^c : A의 여집합, 여자 부분집합
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29%2C%5C%3B%5C%3B%20if%20B%3DA%5Ec%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20A%20%5Ccap%20B%20%3D%20%5Cvarnothing"> 
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20A%5Ec%29%20%3D%20P%28%5COmega%29%20%3D%201%20%3D%20P%28A%29%20&plus;%20P%28A%5Ec%29"> 
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Ctherefore%20P%28A%5Ec%29%3D1-P%28A%29%20%5Cgeq%200%2C%20%5C%3B%5C%3B%20%28kolmogrovs%29%20%5C%3B%5C%3B%200%20%5Cleq%20P%28A%29%20%5Cleq%201">
- 콜모고로프의 공리 1의 조건을 결합하면 모든 확률값은 0과 1사잇값을 가져야 한다.

### 성질3. 포함-배제 원리
- ``포함-배제 원리 inclusion-exclusion principle`` : 두 사건의 합집합의 확률은 각 사건의 확률의 합에서 두 사건의 교집합의 확률을 뺀 것과 같다. (덧셈 규칙 sum rule, addition law)
- <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29%20-%20P%28A%20%5Ccap%20B%29">
- A : 남자 집합, B : 머리 긴 사람 집합
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%20%5Ccup%20%28B%20%5Ccap%20A%5Ec%29%29%20%5C%5C%20%3D%20P%28A%29%20&plus;%20P%28B%20%5Ccap%20A%5Ec%29%20%5C%5C%20%3D%20P%28A%29%20&plus;%20P%28B%20%5Ccap%20A%5Ec%29%20&plus;%20P%28A%20%5Ccap%20B%29%20-%20P%28A%20%5Ccap%20B%29%20%5C%5C%20%3D%20P%28A%29%20&plus;%20P%28%28A%5Ec%20%5Ccap%20B%29%20%5Ccup%20P%28A%20%5Ccap%20B%29%29%20-%20P%28A%20%5Ccap%20B%29%20%5C%5C%20%3D%20P%28A%29%20&plus;%20P%28B%29%20-%20P%28A%20%5Ccap%20B%29">

### 성질4. 전체 확률의 법칙
- 복수의 사건 C_i가 다음을 만족하는 사건들이라고 가정한다.
  - ``배타적 mutually exclusive`` : 서로 교집합이 없다. 배타적 관계
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20C_i%20%5Ccap%20C_j%20%3D%20%5Cvarnothing%20%5C%3B%5C%3B%20%28i%5Cneq%20j%29">
  - ``완전한 부분집합 complete subset`` : 모든 집합의 합집합이 전체집합(표본공간) 이다.  
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20C_1%20%5Ccup%20C_2%20%5Ccup%20%5Ccdots%3D%5COmega">
- ``전체 확률의 법칙 law of total probability`` : 사건 A의 확률은 사건 A와 사건 C_i가 동시에 발생한 사건들의 확률의 합과 같다.
- <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%29%20%3D%20%5Csum_%7Bi%7D%20P%28A%20%5Ccap%20C_i%29"> 

  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20A%20%3D%20A%20%5Ccap%20%5COmega%20%5C%5C%20%3D%20A%20%5Ccap%20%28C_1%20%5Ccup%20C_2%20%5Ccup%20%5Ccdots%29%20%5C%5C%20%3D%20%28A%20%5Ccap%20C_1%29%20%5Ccup%20%28A%20%5Ccap%20C_2%29%20%5Ccup%20%28A%20%5Ccap%20C_3%29%20%5Ccdots">
- C_i 가 서로 공통원소가 없으므로 A∩C_i 도 서로 공통원소가 없다. 따라서,
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%29%20%3D%20%28A%20%5Ccap%20C_1%29%20&plus;%20%28A%20%5Ccap%20C_2%29%20&plus;%20%5Ccdots%20%3D%20%5Csum_%7Bi%7D%20P%28A%20%5Ccap%20C_i%29%3D%5Csum_%7Bi%7D%20P%28A%2CC_i%29">

# 확률분포함수

### 확률분포
- ``확률분포함수 probability distribution`` : 확률이 어디에 어느 정도로 분포되어 있는지를 수학적으로 명시하고 명확하게 전달해주는 도구
    - 어떤 사건에 얼마만큼의 확률이 할당되었는지를 묘사
    - 표본의 수가 많아지면 모든 사건에 대한 확률값을 확인하기 어려워진다.
    - 표본의 수가 무한대이면?
- 종류 : 
    - ``확률질량함수 probability mass function`` : pmf
    - ``누적분포함수 cumulative distribution function`` : cdf
    - ``확률밀도함수 probability density function`` : pdf
    
### 확률질량함수 probability mass function, pmf
- 유한개의 사건이 존재하는 경우 각각의 단순사건에 대한 확률만 정의하는 함수
- ``단순사건 elementary event, atomic event`` : 표본이 하나인 사건 : {1}, {2}, {H}, {T}, {orange}, {남자}, {여자} 등...
    - 단순사건끼리는 서로 교집합이 없으므로 유한개의 사건만 있는 경우 모든 단순사건의 확률값을 알면, 콜모고로프의 공리에 따라서 모든 사건(부분집합)의 확률값을 계산할 수 있다.
    - P({a}) = 0.2, P({b}) = 0.5, P({c}) = 0.3 이면 P({a,b}) = 0.7 
- ``소문자 p를 사용 : p(a) = P({a})``
- 확률과 확률질량함수는 다르다.
    - 확률은 사건을 정의한다. : P({a, b}), P({HH,HT,TT}), P({apple}) 
    - 확률질량함수는 원소만을 가진 단순사건에 대해서만 정의한다. : p(a), p(b), p(1)
    - **따라서 두 개의 원소를 정의할 수 없다. p(1, 2)는 틀린식이다.**
    
### 표본수가 무한하다면?
- 확률질량함수는 **표본 수가 유한한 경우** 각각의 표본에 대해서 확률을 정의하면, 모든 사건(부분집합)에 대해서 확률값을 알 수 있게해준다.
- **표본수가 무한하다면** 확률질량함수를 사용해서 확률을 정의할 수 없다.
- 회전하는 원반에 화살을 쏘고 그 각도를 맞추는 문제
    - 모든 각도가 나올 가능성이 동일하다고 할 떄 모든 각도의 확률을 x라고 한다.
    - 표본의 수는 실수전체이므로 모든 각도의 확률 x와 표본수(무한개)를 곱하면 표본공간의 확률이 된다.
    - 이러한 경우는 표본하나에 대한 사건은 0이어야 한다. 왜냐하면 0이 아니면 표본공간의 확률이 1보다 커지게 되므로 확률을 정의할 수 없다.
        - P({θ=0})=0, P({θ=30}) = 0, P(Ω) = x * 무한대 = 무한대
    - 그러므로 표본수가 무한한 경우에는 사건에 대해서 직접 확률을 정의해야 한다. 
        - P({0 <= θ < 30}) = 1/12, P({30 <= θ < 60}) = 1/12, P({0 <= θ < 60 또는 90 <= θ < 150}) = 1/12 + 1/12 + 1/12 + 1/12 
- 표본수가 무한한 경우의 사건을 확률적으로 정의하기 위해서 확률 함수는 사건(부분집합)을 사용하게 된 것.

### 구간으로 이루어진 사건(부분집합)
- 표본공간이 실수 집합이면 사건(부분집합)은 시작점과 끝점이 있는 **구간**으로 표현된다.
    - A = {a < x <= b}
- 구간(사건, 부분집합)을 입력받아 확률값을 출력하는 함수는 이차원 함수로 표현된다.
    - P(A) = P({a < x <= b}) = P(a, b) : a보다 크고 b보다 작거나 같은 구간의 확률값
- 각각의 구간의 확률값을 정의할 수 있다면 여러 구간으로 이루어진 복잡합 사건의 확률도 콜모고로프의 공리에 따라 구할 수 있다.
    - B = {-2 < x <= 1 or 2 < x <= 3}
    - P(B) = P({-2 < x <= 1 or 2 < x <= 3}) = P({-2 < x <= 1}) + P({2 < x <= 3}) = P(-2, 1) + P(2, 3)

### 누적분포함수
- ``누적분포함수 cumulative distribution function, cdf`` : 구간으로 이루어진 사건의 시작점을 모두 -∞로 맞춘 특수한 구간 S_x에 대한 확률분포
    - S_(-1) = {-∞ < X <= -1} : 음의 무한대에서 -1보다 작거나 같은 구간의 사건
    - S_0 = {-∞ < X <= 0} : 음의 무한대에서 0보다 작거나 같은 구간의 사건
    - S_1 = {-∞ < X <= 1} : 음의 무한대에서 1보다 작거나 같은 구간의 사건
    - S_x = {-∞ < X <= x}
- ``대문자 F(x)로 나타낸다. : F(x) = P(S_x) = P({X <= x})``
    - P({-∞ < X <= x})에서 모든 실수는 음의 무한대보다 큰 값이므로 앞부분을 생략하고 표기한 것.
- 누적분포함수와 콜모고로프의 세 번째 공리를 사용하면 복잡합 구간사건의 확률값을 구할 수 있다. 
    - 콜모고로프의 세 번째 공리 : 두 사건의 교집합이 공집합일때 두 사건 A, B의 합집합의 확률은 각 사건의 확륩값을 더한 것과 같다.
    - <img src="https://latex.codecogs.com/gif.latex?A%20%5Ccap%20B%20%3D%20%5Cvarnothing%2C%20%5C%3B%5C%3B%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29">
    - <img src="https://latex.codecogs.com/gif.latex?P%28-%5Cinfty%2C%20b%29%20%3D%20P%28-%5Cinfty%2C%20a%29%20&plus;%20P%28a%2C%20b%29">
    - <img src="https://latex.codecogs.com/gif.latex?F%28b%29%20%3D%20F%28a%29%20&plus;%20P%28a%2C%20b%29%2C%20%5C%3B%5C%3B%20P%28a%2C%20b%29%20%3D%20F%28b%29%20-%20F%28a%29">
- 누적분포함수의 특징
    - F(-∞) = 0
    - F(+∞) = 1
    - x >= y 이면 F(x) >= F(y) : 단조증가, 0에서 시작하여 1로 서서히 증가한다. 절대로 내려가지 않는다. 
- 회전하는 원반과 화살의 각도를 누적분포함수로 나타내면
    - F(-10) = P({-∞ < θ <= -10}) = 0
    - F(0) = P({-∞ < θ <= 0}) = 0
    - F(10) = P({-∞ < θ <= 10}) = 1/36
    - F(-20) = P({-∞ < θ <= 20}) = 2/36
    - F(350) = P({-∞ < θ <= 350}) = 35/36
    - F(360) = P({-∞ < θ <= 360}) = 36/36
    - F(370) = P({-∞ < θ <= 370}) = 36/36
- 누적분포함수 표현
    - <img src="https://latex.codecogs.com/gif.latex?F%28x%29%20%3D%20P%28S_x%29%20%3D%20P%28%7B-%5Cinfty%20%3C%20X%20%5Cleq%20x%7D%29%20%3D%20P%28-%5Cinfty%2C%20x%29">

### 확률밀도함수
- ``확률밀도함수 probability density function, pdf`` : 누적분포함수의 기울기를 출력하는 도함수로 누적분포함수를 미분하여 구한다.
    - <img src="https://latex.codecogs.com/gif.latex?p%28x%29%20%3D%20%5Cdfrac%20%7BdF%28x%29%7D%7Bdx%7D">
    - 확률밀도함수는 소문자 p 사용
- 누적분포함수의 장점은 복잡한 구간의 시작점을 음의 무한대로 통일해주므로 계산이 간편해진다. 
- 반면에 구간이 크므로 어떤 사건에 확률이 어떻게 분포되어 있는지 알기 어렵다.
- 무한대의 범위를 단위길이 dx로 나누어 단위길이당 확률값을 확인하면 특정 지점에서의 확률의 분포를 확인할 수 있다.
- x_1 근처에서의 단위길에 할당된 확률
    - P({x_1 < x <= x_1 + dx}) = F(x_1 + dx) - F(x_1)
- 같은 구간길이 dx를 가진 x1과 x2의 확률값을 비교해야 하므로, 단위구간 길이당 할당된 확률값
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%20%7BP%28%20%5Cleft%20%5C%7B%20x_1%20%3C%20x%20%5Cleq%20x_1%20&plus;%20dx%20%5Cright%20%5C%7D%29%7D%7Bdx%7D%20%3D%20%5Cdfrac%20%7BF%28x_1%20&plus;%20dx%29%20-%20F%28x_1%29%7D%7Bdx%7D">
- dx를 0으로 줄이면 즉 미세하게 작아지면 누적분포함수의 기울기가 된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Clim_%7Bdx%20%5Crightarrow%200%7D%20%5Cdfrac%20%7BP%28%20%5Cleft%20%5C%7B%20x_1%20%3C%20x%20%5Cleq%20x_1%20&plus;%20dx%20%5Cright%20%5C%7D%29%7D%7Bdx%7D%20%3D%20%5Clim_%7Bdx%20%5Crightarrow%200%7D%20%5Cdfrac%20%7BF%28x_1%20&plus;%20dx%29%20-%20F%28x_1%29%7D%7Bdx%7D">
- 따라서 누적분포함수 그래프의 특정지점의 단위길이당 구간의 확률은 P(x1, x1+dx) = F(x1+dx) - F(x1) 이 되고, 이 값은 미적분학의 기본 원리에 의해서 누적분포함수의 도함수인 확률밀도함수의 해당 구간에서의 정적분의 값 즉 면적과 같다.
    - <img src="https://latex.codecogs.com/gif.latex?F%28x_2%29%20-%20F%28x_1%29%20%3D%20%5Cint_%7Bx_1%7D%5E%7Bx_2%7D%20p%28u%29du">
    - <img src="https://latex.codecogs.com/gif.latex?F%28x_1%29%20%3D%20%5Cint_%7B-%20%5Cinfty%7D%5E%7Bx_1%7D%20p%28u%29du">
    - 누적분포함수 = 원래함수 = 적분함수, 확률밀도함수 = 도함수 = 기울기 함수
    - 누적분포함수 -> 미분 -> 확률밀도함수, 확률밀도함수 -> 정적분 -> 누적분포함수
- 확률밀도함수 pdf의 특징
    - 누적분포함수는 단조증가하므로 도함수인 확률밀도함수의 값은 양수이거나 0과 같다. p(x) >= 0
    - 확률밀도함수의 음의 무한대에서 양의 무한대의 구간의 값은 전체 표본공간의 확률이 되므로 1이다. <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7Dp%28u%29du%3D1">
- **확률도함수의 값은 확률값이 아니다. 특정 구간의 확률이 다른 구간과 비교해서 상대적으로 높은지 낮은지를 비교하는 값이다.**

# python

### 0도에서 180도의 확률이 2배 높은 조작된 원반의 확률을 파이썬으로 구현해보기.
- 시작점과 끝점을 입력받아 확률을 출력하는 함수
- 구간을 입력받아 확률값을 출력하는 함수 : 이차함수와 같다. P(a, b)

```python
def proba_interval(a, b) :
    if  a < 0 : 
        return "wrong start"
    
    if (a < 180) and (b >= 180) :
        if b > a :
            return interval_0_360(a, b)
    
    elif a < 180 :
        if b > a :
            return interval_0_180(a, b)
    
    elif a >= 180 :
        if b > a : 
            return interval_180_360(a, b)
    
def interval_0_360(a, b) :
    print("0_360")
    range_2a = interval_0_180(a, 180)
    range_a = interval_180_360(180, b)
    print(range_2a, range_a)
    return range_2a + range_a
    
def interval_0_180(a, b) :
    print("0_180")
    t = 2/3
    return t / (180/(b-a))
    
def interval_180_360(a, b) :
    print("180_360")
    t = 1/3
    return t / (180/(b-a))

print(proba_interval(30, 90))

=====<print>=====

0_180
0.2222222222222222

print(proba_interval(27, 86))

=====<print>=====

0_180
0.21851851851851853

print(proba_interval(193, 254))

=====<print>=====

180_360
0.11296296296296295

print(proba_interval(90, 270))

=====<print>=====

0_360
0_180
180_360
0.3333333333333333 0.16666666666666666
0.5
```

- 좀 더 간단하게 코딩할 수 있다.
```python

def P(a, b) :
    if a > b :
        raise ValueError('a must be less than b or equal to b')

    a = np.maximum(a, 0)
    b = np.minimum(b, 360)

    if b < 180 :
        return (2 / 3) * ((b - a) / 180)
    else :
        if a < 180 :
            return (2 / 3) * ((180 - a) / 180) + (1 / 3) * ((b - 180) / 180)
        return (1 / 3) * (((b - a) / 180))

P(0, 270)

=====<print>=====

0.8333333333333333
```

- 람다 함수를 사용하면 한 줄로 코딩을 할 수 있다.
- b < 180 이 True 이면 : (2 * (b - a)) / 540 
- b < 180 이 False 이면 : 
    - a < 180 이 True 이면 : (b - 2 * a + 180) / 540
    - a < 180 이 False 이면 : (b - a) / 540
```python
P2 = lambda a, b : (2 * (b - a) if b < 180 else b - 2 * a + 180 if a < 180 else b - a) / 540

P2(0, 270)

=====<print>=====

0.8333333333333334
```

### 0~180도 사이에서 2배 더 잘 박히도록 조작된 원판의 누적분포함수 F(x)를 구하라
- P(-∞, 270) = P(-∞, 180) + P(180, 270)
- F(270) = F(180) + P(180, 270)
    - (2 / 3) + (1 / 3) * ((a - 180) / 180)

```python
def F(a) :
    if a < 0 :
        return 0
    if a > 360 :
        return 1
    elif a < 180 :
        return (2 / 3) * (a / 180)
    elif a >= 180 :
        return (2 / 3) + (1 / 3) * ((a - 180) / 180)

F(270)

=====<print>=====

0.8333333333333333

F(180)

=====<print>=====

0.6666666666666666
```

- 람다 함수 사용
```python

F2 = lambda a : 0 if a < 0 else 1 if a > 360 else (2/3) * (a / 180) if a < 180 else (2 / 3) + (1 / 3) * ((a - 180) / 180)
F3 = lambda a : 0 if a < 0 else 1 if a > 360 else a / 270 if a < 180 else (2 / 3) + ((a - 180) / 540)

F2(270)

=====<print>=====

0.8333333333333333

F3(270)

=====<print>=====

0.8333333333333333
```
