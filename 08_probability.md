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
- 확률이라는 함수를 정의한다는 것은 "A가 선택될 확률이 얼마인가?"라는 질문에 대한 답을 모든 경우(사건, 부분집합)에 대해서 **미리 준비해 놓은 것** 또는 **할당해 놓은 것**과 같다.

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
  > <img src="https://latex.codecogs.com/gif.latex?P%28%5Cvarnothing%29%20%3D%200"> \
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20P%28A%20%5Ccup%20B%29%20%3D%20P%28A%29%20&plus;%20P%28B%29%2C%20%5C%3B%5C%3B%20if%20%5C%3B%5C%3B%20P%28B%29%20%3D%20%5Cvarnothing%2C%5C%3B%20A%20%5Ccup%20%5Cvarnothing%20%3D%20A%20%5Crightarrow"> \
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5C%3B%5C%3B%20P%28A%20%5Ccup%20%5Cvarnothing%29%20%3D%20P%28A%29%20%3D%20P%28A%29%20&plus;%20P%28%5Cvarnothing%29"> \
  > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Ctherefore%20P%28%5Cvarnothing%29%20%3D%200">































