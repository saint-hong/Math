# 엔트로피
- 확률론에서의 엔트로피
    - 확률분포의 모양을 설명하는 특정값이다.
    - 확률분포가 가지고 있는 정보의 양을 나타내는 값이다.
    - 두 확률분포의 모양이 어떤 관계를 갖는지 또는 유사한 정도를 표현한다.
    - **조건부엔트로피**는 한 확률분포에 대해서 다른 확률분포가 받는 영향을 설명해준다.
    - **교차엔트로피와 쿨벡-라이블러 발산**은 두 확률분포가 얼마나 닮았는지를 나타낸다.
    - **상호정보량**은 두 확률분포의 독립 및 상관관계를 나타낸다. 

## 1. 엔트로피
- `엔트로피 entropy` : 확률분포의 이러한 정보의 차이를 하나의 숫자로 나타내 준다.
- Y=0, Y=1인 두 가지 값을 갖는 두 확률변수의 세가지 확률분포
    - Y_1 : P(Y=0) = 0.5, P(Y=1) = 0.5
    - Y_2 : P(Y=0) = 0.8, P(Y=1) = 0.2
    - Y_3 : P(Y=0) = 1.0, P(Y=1) = 0.0
- 베이지안 확률론 관점에서 위 확률분포에서 얻을 수 있는 정보
    - Y_1은 y값에 대해서 아무것도 모르는 상태와 같다.
    - Y_2는 y=0이라고 믿지만 아닐 가능성도 있다는 것을 아는 상태와 같다.
    - Y_3은 y=0이라고 100% 확신하는 상태이다.

## 1-1. 엔트로피의 정의
- `엔트로피 entropy` : 확률분포가 갖는 **정보의 확신도** 혹은 **정보량**을 수치로 표현한 것.
- 엔트로피값의 의미
    - 어떤 확률분포에서 특정한 값이 나올 확률이 높아지고 다른 값이 나올 확률이 낮아지면 엔트로피값이 작아진다.
    - 여러가지 값이 나올 확률이 비슷하다면 엔트로피값은 높아진다.
    - 확률분포의 모양에 대한 특성값
    - 확률또는 확률밀도가 특정한 값에 몰려있으면 엔트로피가 작고, 반대로 여러값에 퍼져있으면 엔트로피가 크다고 할 수 있다.

#### 물리학의 엔트로피 용어를 사용한 것
- 물리학에서의 엔트로피는 물질의 상태가 분산되는 정도를 나타낸다. 물체의 상태가 고루 분산되어 있으면(무질서하면) 엔트로피가 높고, 특정한 하나의 상태로 몰려있으면 엔트로피가 낮다.
- 물리학에서의 엔트로피는 "무질서의 정도"의 의미로 해석되기도 한다.
    - 우주는 엔트로피가 낮은 쪽에서 높은 곳으로 활동한다.

#### 엔트로피의 수학적 의미
- 엔트로피는 확률분포함수를 입력으로 받아 숫자를 출력하는 **범함수 functional**와 같다.
- 범함수의 표기 방법에 따라서 "H[ ]" 이렇게 표기 한다.
- 확률변수 Y가 카테고리분포와 같은 이산확률변수인 경우 엔트로피
    - $\text{H}[Y] = - \sum_{k=1}^{K} p(y_k) log_2 p(y_k)$
    - K : X가 가질 수 있는 클래스(범주값)의 수
    - p(y_k) : 확률질량함수(pmf)
    - 확률의 로그값은 항상 음수이므로 -를 곱하여 양수값으로 만들어 준다.
    - 확률은 0과 1사이의 값이고 로그함수에서 0과 1 사이의 영역은 음수값이다.
- 확률변수 Y가 정규분포와 같은 연속확률변수인 경우 엔트로피
    - $\text{H}[Y] = - \int_{- \infty}^{\infty} p(y) log_2 p(y) dy$
    - p(y) : 확률밀도함수(pdf)
    - log2 : 정보통신 분야에서의 로그값 사용관례
- 엔트로피 계산에서 p(y) = 0 인 경우는 로그값이 정의되지 않는다. 따라서 로피탈의 정리(l'Hopitals rule)를 따라서 0으로 수렴하는 값으로 계산한다.
    - $\lim_{p \rightarrow 0} p log_2 p = 0$

#### Y_1, Y_2, Y_3의 이산확률분포에 대한 엔트로피
- $\text{H}[Y_1] = - \dfrac{1}{2} log_2 \dfrac{1}{2} - \dfrac{1}{2} log_2 \dfrac{1}{2} = 1$
    - -1log_2 1/2 = log_2 1/2^-1 = log_2 2 = 1
- $\text{H}[Y_2] = - \dfrac{8}{10} log_2 \dfrac{8}{10} - \dfrac{2}{10} log_2 \dfrac{2}{10} \approx 0.72$
    - (- 8/10 * np.log2(8/10)) - (2/10 * np.log2(2/10)) = 0.7219280948873623
- $\text{H}[Y_3] = - 1 log_2 1 - 0 log_2 0 = 0$

### [python] Y_1, Y_2, Y_3의 확률분포
- 엔트로피의 크기 순서 : 확률이 골고루 분포되어있으면 엔트로피가 크고, 확률이 한쪽에 몰려있으면 엔트로피가 작다.
    - Y_1 > Y_2 > Y_3

```python
%matplotlib inline

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar([0, 1], [0.5, 0.5])
plt.xticks([0, 1], ["Y=0", "Y=1"])
plt.ylim(0, 1.1)
plt.title("$Y_1$")

plt.subplot(132)
plt.bar([0, 1], [0.8, 0.2])
plt.xticks([0, 1], ["Y=0", "Y=1"])
plt.ylim(0, 1.1)
plt.title("$Y_2$")

plt.subplot(133)
plt.bar([0, 1], [1.0, 0.0])
plt.xticks([0, 1], ["Y=0", "Y=1"])
plt.ylim(0, 1.1)
plt.title("$Y_3$")

plt.tight_layout()
plt.show() ;
```

![ent_1.png](./images/entropy/ent_1.png)

### [python] 넘파이로 엔트로피 계산

```python
y_1 = - 0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)
y_1

>>>

1.0
```

```python
y_2 = - 0.8 * np.log2(8/10) - 0.2 * np.log2(2/10)
y_2

>>>

0.7219280948873623
```

```python
eps = np.finfo(float).eps
y_3 = - 1 * np.log2(1) - eps * np.log2(eps)
y_3

>>>

1.1546319456101628e-14
```

### [python] quiz
- 베르누이분포에서 확률값 P(Y=1)은 0부터 1까지의 값을 가질 수 있다. 각각의 값에 대해 엔트로피를 계산하여 가로축이 P(Y=1)이고 세로축이 H[Y]인 그래프를 그리시오.
- 베르누이분포는 입력값이 0 또는 1이므로 P(Y=1)인 경우의 엔트로피와 P(Y=0)인 경우의 엔트로피를 함께 계산해야 한다.
    - 즉 P(Y=1) = 0.3 이면, P(Y=0) = 0.7 이다.

```python
def entropy(x) :
    a = x
    b = 1-x
    return - a * np.log2(a) - (b) * np.log2(b)

xx = np.linspace(0.001, 1-0.001, 1000)

plt.figure(figsize=(8, 6))
plt.plot(xx, entropy(xx), 'b-')
plt.xticks(np.arange(0, 11)/10)
plt.xlabel("$P(Y=1)$")
plt.ylabel("entropy")
plt.title("베르누이 분포 Y_1의 엔트로피")
plt.show() ;
```

![ent_2.png](./images/entropy/ent_2.png)

### [풀이]

```python
plt.figure(figsize=(8, 6))
P0 = np.linspace(0.001, 1-0.001, 1000)
P1 = 1 - P0
H = - P0 * np.log2(P0) - P1 * np.log2(P1)

plt.plot(P1, H, '-', label='엔트로피')
plt.xlabel('P(Y=1)')
plt.ylabel("entropy")
plt.legend()
plt.show() ;
```

![ent_3.png](./images/entropy/ent_3.png)

### [python] quiz
- 다음 확률분포의 엔트로피를 계산하시오.
    - P(Y=0) = 1/8, P(Y=1) = 1/8, P(Y=2) = 1/4, P(Y=3) = 1/2
    - P(Y=0) = 1, P(Y=1) = 0, P(Y=2) = 0, P(Y=3) = 0
    - P(Y=0) = 1/4, P(Y=1) = 1/4, P(Y=2) = 1/4, P(Y=3) = 1/4

```python
Y0 = - 1/8 * np.log2(1/8) - 1/8 * np.log2(1/8) - 1/4 * np.log2(1/4) - 1/2 * np.log2(1/2)
Y0

>>>

1.75
```

```python
eps = np.finfo(float).eps
Y1 = - 1 * np.log2(1) - eps * np.log2(eps) - eps * np.log2(eps) - eps * np.log2(eps)
Y1

>>>

3.4638958368304884e-14
```

```python
Y2 = - 1/4 * np.log2(1/4) - 1/4 * np.log2(1/4) - 1/4 * np.log2(1/4) - 1/4 * np.log2(1/4)
Y2

>>>

2.0
```

## 1-2 엔트로피의 성질
- **엔트로피의 최소값** : 확률변수가 결정론적이면 확률분포에서 어떤 값이 나올 확률은 1과 같다. 따라서 이러한 경우 엔트로피는 0이다. 이 값이 엔트로피가 가질 수 있는 최소값이다.
- **엔트로피의 최대값** : 이산확률변수의 클래스의 개수에 따라서 엔트로피의 값이 달라진다. 이산확률분포가 가질 수 있는 클래스의 개수가 2^K개일때 엔트로피의 최대값은 K이다.
    - $H = - 2^K \cdot \dfrac{1}{2^K} log_2 \dfrac{1}{2^K} = K$
    - 만약 이산확률분포의 클래스의 개수가 2^3=8이면 위의 계산에 의해 엔트로피가 가질 수 있는 최대값은 3이 된다.

## 1-3 엔트로피의 추정
- 이론적인 확률밀도함수가 없는 경우, 실제 데이터로부터 확률질량함수를 추정하여 엔트로피를 계산한다.
    - **데이터 -> 확률질량함수 추정 -> 엔트로피 계산**
- 전체 데이터 수 80개, Y=0인 데이터가 40개, Y=1인 데이터가 40개 있는 경우의 엔트로피
    - $\text{H}[Y] = - \dfrac{40}{80} log_2(\dfrac{40}{80}) - \dfrac{40}{80} log_2(\dfrac{40}{80}) = \dfrac{1}{2} + \dfrac{1}{2} = 1$

### [python] 사이파이로 entropy 계산

```python
p = [0.5, 0.5]
sp.stats.entropy(p, base=2)

>>>

1.0
```

```python
p = [0.3, 0.7]
sp.stats.entropy(p, base=2)

>>>

0.8812908992306927
```

```python
p = [0.8, 0.2]
sp.stats.entropy(p, base=2)

>>>

0.7219280948873623
```

### [python] quiz
- 데이터가 모두 60개이고 그 중 Y=0인 데이터가 20개, Y=1인 데이터가 40개 있는 경우 엔트로피를 계산하시오
- 데이터가 모두 40개이고 그 중 Y=0인 데이터가 30개, Y=1인 데이터가 10개 있는 경우 엔트로피를 계산하시오
- 데이터가 모두 20개이고 그 중 Y=0인 데이터만 20개 있는 경우의 엔트로피를 계산하시오.

```python
p0 = [20/60, 40/60]
sp.stats.entropy(p0, base=2)

>>>

0.9182958340544894
```

```python
p1 = [30/40, 10/40]
sp.stats.entropy(p1, base=2)

>>>

0.8112781244591328
```

```python
p2 = [20/20, eps]
sp.stats.entropy(p2, base=2)

>>>

1.1866662106483117e-14
```

## 1-4 가변길이 인코딩
- **엔트로피는 원래 통신 분야에서 데이터가 가지고 있는 정보량을 계산하기 위해 고안되었다.**
- A, B, C, D 4글자로 이루어진 문서가 있다.
- 각 글자를 0, 1로 이루어진 이진수로 변환하면 다음과 같다.
    - A = "00"
    - B = "01"
    - C = "10"
    - D = "11"
- ABCD로 이루어진 문서가 200글자이면 인진수는 400개가 된다. 문서의 길이가 길어지면 이진수의 갯수는 더 많아지게 된다. 문자를 이진수로 변환할 때 이진수의 수를 줄이기 위한 방법으로 **가변길이 인코딩 variable length encoding**을 사용한다.
- 가변길이 인코딩은 ABCD 각 문자가 문서에 등장한 빈도수에 따라서 변환할 인코딩 문자의 길이를 다르게 할 수 있다.
- 어떤 문서에서 ABCD 각 문자의 빈도수 분포를 조사했을 때 다음과 같다.
    - P(Y=A) = 1/2, P(Y=B) = 1/4, P(Y=C) = 1/8, P(Y=D) = 1/8
    - A가 가장 많이 등장하고 다음으로 B가 많이 등장하고 C와 D는 빈도수가 같다.
- 이러한 경우 가장 많이 나오는 A의 인코딩 문자의 갯수를 줄이고 C와 D의 인코딩 문자의 갯수를 늘일 수 있다.
- 가변길이 인코딩 방식으로 인코딩 된 문서의 한 글자당 비트수를 계산하면 엔트로피의 값과 같다.
    - $\left( 200 \text{X} \dfrac{1}{2} \right) \cdot 1 + \left( 200 \text{X} \dfrac{1}{4} \right) \cdot 2 + \left( 200 \text{X} \dfrac{1}{8} \right) \cdot 3 + \left( 200 \text{X} \dfrac{1}{8} \right) \cdot 3 = 350$
    - **한 문자당 바이트 수 : 350 / 200 = 1.75**
    - 각 문자의 확률에 대한 엔트로피 :
$H=-\dfrac{1}{2}log_2\dfrac{1}{2} - \dfrac{1}{4}log_2\dfrac{1}{4} - \dfrac{1}{8}log_2\dfrac{1}{8} - \dfrac{1}{8}log_2\dfrac{1}{8}=1.75$

### [python] 가변길이 인코딩

#### ABCD로 조합된 문서 만들기

```python
N = 200
p = [1/2, 1/4, 1/8, 1/8]
doc0 = list("".join([int(N * p[i]) * c for i, c in enumerate("ABCD")]))
np.random.shuffle(doc0)
doc = "".join(doc0)
doc

>>>

'BAADDAABAAAAAADAACDBCBDBBAABCCDBADABDDAADAAAABBAABBAACDAACABCAAABAABADAABBBDBACCAAADACADCABABACBCABADACAAAAAAAABBAABAADAAABBBBBACCDCAAAABABBACDBBBDBAABBAAAADCAACDAAABACABBBCBDBCAAAAACAAABAADDAAACBBAAA'
```

#### 코드 풀이
- ABCD의 각 확률대로 만든다.

```python
[int(N * p[i]) * c for i, c in enumerate("ABCD")]

>>>

['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
 'CCCCCCCCCCCCCCCCCCCCCCCCC',
 'DDDDDDDDDDDDDDDDDDDDDDDDD']
```

- join 함수로 합쳐준다.

```python
"".join([int(N * p[i]) * c for i, c in enumerate("ABCD")])

>>>

'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCCCCCCCCCCDDDDDDDDDDDDDDDDDDDDDDDDD'
```

- list로 변환하면 각 문자가 하나의 str 원소로 바뀐다.

```python
print(list("".join([int(N * p[i]) * c for i, c in enumerate("ABCD")])))

>>>

['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
```

- np.random.shuffle로 원소를 무작위로 섞어준 후, str 원소를 join으로 다시 합쳐준다.

```python
test = list("".join([int(N * p[i]) * c for i, c in enumerate("ABCD")]))
np.random.shuffle(test)
test = "".join(test)
test

>>>

'ACBBAAAABAAAAAADACBABCDBCAAACAACBCAABDDAADCCABBABBAACAAAAABDDBACDAABBBAAACBCABDACDDABBBADABDBBACBAAABAAAAAAAAABAABCABBBDCBABBAAABAACCABDABADADCDABCABAAAADAADBAAABABDDCABBABAACAAABADCAAABAAADCAAAABBDAA'
```

#### 이 문서를 이진수로 인코딩 하기

```python
encoder = {"A" : "00", "B" : "01", "C" : "10", "D" : "11"}
encoded_doc = "".join([encoder[c] for c in doc])
encoded_doc

>>>

'0100001111000001000000000000110000101101100111010100000110101101001100011111000011000000000101000001010000101100001000011000000001000001001100000101011101001010000000110010001110000100010010011000010011001000000000000000000101000001000011000000010101010100101011100000000001000101001011010101110100000101000000001110000010110000000100100001010110011101100000000000100000000100001111000000100101000000'
```

#### 이진수로 인코딩하면 400개의 문자가 된다.
- 원래 문서는 길이가 200이었다.

```python
len(encoded_doc)

>>>

400
```

#### 인코딩한 문서의 문자별 빈도수를 분포로 확인
- A > B > C = D 의 순으로 사용 횟수가 다르다.
- sns.countplot(list(doc), order="ABCD") : list의 원소를 order의 인수로 입력하면 자동으로 count해서 bar로 표현해준다.

```python
plt.figure(figsize=(8, 6))
sns.countplot(list(doc), order="ABCD")
plt.title("글자수의 분포 : ABCD")
plt.show() ;
```
![ent_4.png](./images/entropy/ent_4.png)

#### 가변길이 인코딩 방식을 사용하여 각 문자별 인코딩 문자의 갯수 조정
- 많이 등장한 문자는 인코딩 문자를 줄이고, 적게 등장한 문자는 인코딩 문자를 늘려준다.
- 즉 처음 인코딩한 문서의 길이보다 줄어든다.

```python
v1_encoder = {"A" : "0", "B" : "10", "C" : "110", "D" : "111"}
v1_encoded_doc = "".join([v1_encoder[c] for c in doc])
v1_encoded_doc

>>>

'10001111110010000000111001101111011010111101000101101101111001110101111110011100001010001010001101110011001011000010001001110010101011110011011000011101100111110010010011010110010011101100000000010100010001110001010101010011011011111000001001010011011110101011110001010000011111000110111000100110010101011010111101100000011000010001111110001101010000'
```
- 가변길이 인코딩으로 변환한 문서의 길이

```python
len(v1_encoded_doc)

>>>

350
```

#### 가변길이 인코딩 된 문서를 한 글자당 인코딩 비트 수로 나누면?
- 각 문자의 확률에 대한 엔트로피 값과 같다.

```python
350/200

>>>

1.75

p = [1/2, 1/4, 1/8, 1/8]
sp.stats.entropy(p, base=2)

>>>

1.75
```

### [python] quiz
- A, B, C, D, E, F, G, H 의 8개 글자로 이루어진 문서가 있을 때 각각의 글자가 나올 확률이 다음과 같다.
- 이 문서를 위한 가변길이 인코딩 방식을 서술하고 한 글자를 인코딩하는데 필요한 평균 비트수를 계산하라.

#### 평균비트수 = 엔트로피

```python
p_docs = [1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64]
sp.stats.entropy(p_docs, base=2)

>>>

2.0
``

## 1-6 엔트로피의 최대화
- 기대값=0, 분산=sigma^2 이 주어진 경우 엔트로피 H[p(x)]를 가장 크게 만드는 확률밀도함수 p(x)는 정규분포가 된다.
- 정규분포는 기댓값과 표준편차를 알고있는 확률분포들 중에서 엔트로피가 가장 크다.
    - 따라서 가장 정보가 적은 확률분포이기도 하다.
    - 정규분포는 **무정보 사전확률분포**로서 베이즈추정의 사전분포에 사용되는 경우가 많다.
#### 증명 
- 목적범함수인 엔트로피의 값을 최대화한다. 이 과정에서 입력변수인 확률밀도함수의 제한조건이 라그랑주 승수법으로 추가된다. 엔트로피함수에 제한조건이 추가되어 새로운 목적함수가 된다. 이 목적함수를 풀면 정규분포의 형태가 된다. 
- **pdf의 제한조건 1 : 면적의 총합은 1이다.**
    - $\int_{-\infty}^{\infty} p(x)dx = 1$
- **pdf의 제한조건 2 : 기대값은 0이다. (연속확률변수의 기대값 공식)**
    - $\int_{-\infty}^{\infty} xp(x)dx = 0$
- **pdf의 제한조건 3 : 분산은 sigma^2 이다.**
    - $\int_{-\infty}^{\infty} x^2p(x)dx = \sigma^2$
- **목적범함수 objective functional**인 엔트로피를 최대화 한다. 
    - $\text{H}[Y] = - \int_{-\infty}^{\infty}p(x)logp(x)dx$
- 목적범함수에 pdf의 제한조건 3가지를 라그랑주 승수법으로 추가한다. 
$\begin{aligned}
\text{H}[Y]
&= - \int_{-\infty}^{\infty}p(x)\log p(x)dx + \lambda_1 \left( \int_{-\infty}^{\infty} p(x)dx -1 \right) \\
&+ \lambda_2 \left( \int_{-\infty}^{\infty} xp(x)dx \right) + \lambda_3 \left( \int_{-\infty}^{\infty} x^2p(x)dx - \sigma^2 \right) \\
&= \int_{-\infty}^{\infty} ( -p(x)\log p(x) + \lambda_1p(x) + \lambda_2xp(x) + \lambda_3x^2p(x) - \lambda_1 - \lambda_3\sigma^2) dx\\
\end{aligned}$
- 최대값을 구하기위해서 목적함수를 확률밀도함수로 편미분한다.
    - $\dfrac{\delta H}{\delta p(x)} = - logp(x) - 1 + \lambda_1 + \lambda_2x + \lambda_3x^2 = 0$
- 양변에 지수함수를 적용하여 정리하면 다음과 같아진다. 
    - $p(x) = exp( - 1 + \lambda_1 + \lambda_2x + \lambda_3x^2  )$
- pdf의 제한조건 3가지와 이 제한조건을 만족하는 연립방정식을 계산하면 라그랑주 승수 3개가 나온다. (이 과정은 생략함)
    - $\lambda_1 = 1 - \dfrac{1}{2} \log 2 \pi \sigma^2$
    - $\lambda_2 = 0$
    - $\lambda_3 = - \dfrac{1}{2\sigma^2}$
- 라그랑주 승수값을 각각 위의 식에 대입하면 정규분포의 확률밀도함수가 된다.
    - $p(x) = \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp \left(- \dfrac{x^2}{2\sigma^2} \right)$
    
#### 정규분포의 확률밀도함수에 대한 엔트로피
- $\text{H}[p(x)] = \dfrac{1}{2}\left( 1 + \log 2 \pi \sigma^2 \right)$

## 1-7 조건부 엔트로피
- 두 확률변수의 결합엔트로피(joint entropy)와 조건부엔트로피(conditional entropy)를 구하고 분류문제에 적용한다. 

### 결합엔트로피
- `결합엔트로피 joint entropy` : 결합확률분포를 사용하여 정의한 엔트로피이다.
- 이산확률변수 : $\text{H}[X, Y] = - \sum_{i=1}^{K_x} \sum_{j=1}^{K_Y} p(x_i, y_i)\log_2 p(x_i, y_i)$
    - K_X, K_Y 는 X의 갯수, Y의 갯수
    - p(x)는 확률질량함수
- 연속화률변수 : $\text{H}[X, Y] = - \int_{x} \int_{y} p(x, y) \log_2 p(x, y) dx dy$
    - p(x)는 확률밀도함수
- 결합엔트로피도 엔트로피와 같다. 확률분포가 고르게 퍼져있으면 엔트로피 값이 증가하고, 확률분포가 하나의 값에 치우쳐져 있으면 엔트로피 값이 감소한다.     
    
### 조건부엔트로피
- `조건부엔트로피 conditioanl entropy` : 어떤 확률변수 X가 다른 확률변수 Y의 값을 예측하는데 도움이 되는지를 측정하는 방법이다. 
- 확률변수 X가 하나의 값을 갖고 확률변수 Y도 하나의 값만 갖는다면 X로 Y를 예측할 수 있다.
- 확률변수 X가 하나의 값을 갖고 확률변수 Y가 여러개의 값에 퍼져있다면 X로 Y를 예측할 수 없다. 
- 수학적 정의
    - $\text{H}[Y|X=x_i] = - \sum_{j=1}^{K_Y} p(y_j | x_i) \log_2 p(y_j | x_i)$
- 조건부엔트로피는 확률변수 X가 가질 수 있는 모든 경우에 대해 H[Y|X=x_i]를 가중평균한 값과 같다.
- 이산확률변수
    - $\text{H}[Y|X] = \sum_{i=1}^{K_x} p(x_i) \text{H}[Y|X=x_i] = - \sum_{i=1}^{K_x} \sum_{j=1}^{K_Y} p(x_i, y_i) \log_2 p(y_j | x_i)$
- 연속확률변수
    - $\text{H}[Y|X] = - \int_{y} p(y|x) \log_2 p(y|x) dy = - \int_{x} \int_{y} p(x, y) \log_2 p(y|x) dx dy$

### [python] 조건부엔트로피, 결합 엔트로피

#### 확률변수 X가 확률변수 Y의 예측에 도움이 되는 경우
- **X가 특정한 값일 때 Y도 특정한 값이 될때 예측에 도움이 된다.**
- 두 확률변수 X, Y의 관계가 다음과 같다.

```
    | Y=0 | Y=1
----------------
X=0 | 0.4 | 0.0
----------------
X=1 | 0.0 | 0.6
```
- 조건부확률분포 : conditional = joint / marginal
    - P(Y=0|X=0) = P(X=0, Y=0) / P(X=0) = 1
    - P(Y=1|X=0) = P(X=0, Y=1) / P(X=0) = 0
    - P(Y=0|X=1) = P(X=1, Y=0) / P(X=1) = 0
    - P(Y=1|X=1) = P(X=1, Y=1) / P(X=1) = 1
- Y의 엔트로피는 모두 0이다.
    - H[Y|X=0] = - 1 log2(1) - 0 log2(0) = 0
    - H[Y|X=1] = - 0 log2(0) - 1 log2(1) = 0
- 조건부 엔트로피도 0이 된다.
    - H[Y|X] = 0

- X=0 이면 Y=0 이다.
- X=1 이면 Y=1 이다.

```python
plt.figure(figsize=(8, 5))
ax1 = plt.subplot(121)
pXY = [[0.4, 0], [0, 0.6]]
sns.heatmap(pXY, annot=True, cbar=False)
plt.xlabel("Y")
plt.ylabel("X")

plt.subplot(222)
plt.bar([0, 1], [1.0, 0])
plt.ylim(0, 1)
plt.title("조건부확률분포 p(Y|X=0)")

plt.subplot(224)
plt.bar([0, 1], [0, 1.0])
plt.ylim(0, 1)
plt.title("조건부확률분포 p(Y|X=1)")

plt.suptitle("조건부 엔트로피 H[Y|X]=0")
plt.tight_layout()
plt.show() ;
```

![ent_5.png](./images/entropy/ent_5.png)

#### 확률변수 X가 확률변수 Y의 예측에 도움이 되지 않는 경우
- **X가 특정한 값일때 Y는 여러값의 분포가 생길 때 예측에 도움이 되지 않는다.**
- 두 확률변수 X, Y의 관계가 다음과 같다.

```
    | Y=0 | Y=1
----------------
X=0 | 1/9 | 2/9
----------------
X=1 | 2/9 | 4/9
```
- 조건부확률분포 : conditional = joint / marginal
    - P(Y=0|X=0) = P(X=0, Y=0) / P(X=0) = 1/3 
    - P(Y=1|X=0) = P(X=0, Y=1) / P(X=0) = 2/3
    - P(Y=0|X=1) = P(X=1, Y=0) / P(X=1) = 1/3
    - P(Y=1|X=1) = P(X=1, Y=1) / P(X=1) = 2/3
- Y의 엔트로피는 0.92에 가깝다.
    - H[Y|X=0] = H[Y|X=1] = - 1/3 log2(1/3) - 2/3 log2(2/3) = 0.92
- 가중평균한 조건부엔트로피도 0.92에 가깝다.
    - H[Y|X] = 1/3 H[Y|X=0] + 2/3 H[Y|X=1] = 0.92 

#### Y의 엔트로피

```python
sp.stats.entropy([1/3, 2/3], base=2)

>>>

0.9182958340544894
```

```python
pXY = [[1/9, 2/9], [2/9, 4/9]]

plt.figure(figsize=(8, 5))
ax1 = plt.subplot(121)
sns.heatmap(pXY, annot=True, cbar=False)
plt.xlabel("Y")
plt.ylabel("X")

plt.subplot(222)
plt.bar([0, 1], [1/3, 2/3])
plt.ylim(0, 1)
plt.title("조건부확률분포 p(Y|X=0)")

plt.subplot(224)
plt.bar([0, 1], [1/3, 2/3])
plt.ylim(0, 1)
plt.title("조건부확률분포 p(Y|X=1)")

plt.suptitle("조건부엔트로피 H[Y|X]=0.92")
plt.tight_layout()
plt.show() ;
```

![ent_6.png](./images/entropy/ent_6.png)

### [python] 조건부엔트로피를 사용한 스팸메일 분류문제
- 학습용 메일 80개 중 정상메일 40개 (Y=0), 스팸메일 40개 (Y=1)이 있다고 가정한다.
- 스팸메일인지 아닌지 구분하기 위해 특정 키워드가 존재하면 X=1, 키워드가 존재하지 않으면 X=0으로 정의한다.
- 키워드로 X1, X2, X3이 있다.

#### 키워드 X1과 Y의 관계
```
     | Y=0 | Y=1
---------------------
X1=0 | 30  | 10  | 40
---------------------
X1=1 | 10  | 30  | 40
---------------------
       40  | 40  | 80
```

#### 키워드 X2과 Y의 관계
```
     | Y=0 | Y=1
---------------------
X2=0 | 20  | 40  | 60
---------------------
X2=1 | 20  |  0  | 20
---------------------
       40  | 40  | 80
```

#### 키워드 X3과 Y의 관계
```
     | Y=0 | Y=1
---------------------
X3=0 |  0  | 40  | 40
---------------------
X3=1 | 40  |  0  | 40
---------------------
       40  | 40  | 80
```

- 스팸메일인지 아닌지 확인할 수 있는 키워드는 X3이다. X3이 있는 경우와 없는 경우의 정상메일, 스팸메일의 구분이 명확하다.
- X1과 X2 중에서는 어떤 키워드가 스팸메일을 구분하기에 더 좋을까?

#### **조건부엔트로피값을 사용하여 구할 수 있다.**
- X1과 Y의 조건부엔트로피
    - H[Y|X1]=p(X1=0)H[Y|X1=0] + p(X1=1)H[Y|X1=1] = 40/80x0.81 + 40/80x0.81 = 0.81
    - H[Y|X1=0]와 H[Y|X1=1]는 각각 따로 구할 수 있다.

- X2와 Y의 조건부엔트로피
    - H[Y|X2]=p(X2=0)H[Y|X2=0] + p(X2=1)H[Y|X2=1] = 60/80x0.92 + 20/80x0 = 0.69

- X3와 Y의 조건부엔트로피
    - H[Y|X3]=p(X3=0)H[Y|X3=0] + p(X3=1)H[Y|X3=1] = 0

- `X2가 X1 보다 좋은 키워드 이다.` : 엔트로피값이 낮다는 것에 의하여, X2의 값이 0 또는 1일때 Y가 정상인지 스팸인지 한쪽값에 확률이 더 쏠린다는 것을 알 수 있다.

#### 조건부엔트로피는 의사결정나무에 사용된다.
- `의사결정나무 decision tree` 분류모형은 조건부엔트로피를 사용하여 가장 좋은 특징값과 기준점을 찾아준다.

### [python] 조건부엔트로피를 사용한 붓꽃 분류문제
- 붓꽃 데이터에서 버지니카와 베르시칼라 품종을 꽃받침의 길이(sepal length)로 분류하려고 한다.
- 꽃받침의 길이 중 특정한 값을 기준으로 품종의 갯수를 구한뒤, 조건부엔트로피를 계산한다.
    - 조건부엔트로피 값이 작을 수록 X로 Y를 예측하는데 더 도움이 된다는 의미이다.
    - 즉 6cm 가 6.5cm 보다 베르시칼라와 버지니카 품종을 구분하는데 더 도움이 된다.

#### 버지니카와 베르시칼라 품종만 임포트

```python
from sklearn.datasets import load_iris

iris = load_iris()
idx = np.in1d(iris.target, [1, 2])
X = iris.data[idx, :]
y = iris.target[idx]

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target[idx]
df.tail() ;
```

![ent_7.png](./images/entropy/ent_7.png)

#### 확률밀도 확인
- 기준점을 어떻게 잡아야 두 품종을 구분할 수 있을까?
- 조건부엔트로피 값이 작은 기준점을 찾는다.

```python
plt.figure(figsize=(8, 6))

sns.distplot(df[df.species==1]["sepal length (cm)"], hist=True, rug=True, label="버지니카")
sns.distplot(df[df.species==2]["sepal length (cm)"], hist=True, rug=True, label="베르시칼라")
plt.xlabel("꽃받침의 길이")
plt.title("꽃받침의 길이와 붓꽃의 품종")
plt.legend()
plt.show() ;
```

![ent_8.png](./images/entropy/ent_8.png)

#### 6cm 로 기준을 잡았다면?
- 6보다 큰 것과 6보다 작은 것 중 각 품종의 갯수

```python
df["x1"] = df["sepal length (cm)"] > 6
pivot_table1 = df.groupby(["x1", "species"]).size().unstack().fillna(0)
pivot_table1
```

![ent_9.png](./images/entropy/ent_9.png)

#### 조건부 엔트로피 계산

```python
def cond_entropy(v) :
    # conditional = joint / marginal
    pYX0 = v[0, :] / np.sum(v[0, :])
    pYX1 = v[1, :] / np.sum(v[1, :])
    HYX0 = sp.stats.entropy(pYX0, base=2)
    HYX1 = sp.stats.entropy(pYX1, base=2)

    # 조건부엔트로피는 H[Y|X=xi]를 가중평균한 값과 같다.
    HYX = np.sum(v, axis=1) @ [HYX0, HYX1] / np.sum(v)

    return HYX
```

#### 6cm 로 기준을 잡은 경우의 조건부 엔트로피

```python
cond_entropy(pivot_table1.values)

>>>

0.860714271586387
```

#### 6.5cm 를 기준으로 구분하는 경우

```python
df["x2"] = df["sepal length (cm)"] > 6.5
pivot_table2 = df.groupby(["x2", "species"]).size().unstack()
pivot_table2
```

![ent_10.png](./images/entropy/ent_10.png)

```python
cond_entropy(pivot_table2.values)

>>>

0.9306576387006182
```

#### 기준값이 6cm 엔트로피 값이 더 작으므로 6cm를 기준으로 삼는 것이 더 좋다.

### [python] quiz

#### - 붓꽃데이터에서 꽃받침의 길이 sepal length의 최소값과 최대값 구간을 0.05 간격으로 나누어 각각의 값을 기준값으로 했을 때 조건부엔트로피가 어떻게 변하는지 그래프로 그리시오.

- 붓꽃 데이터로 데이터 프레임 생성

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
df_ir = pd.DataFrame(X, columns=iris.feature_names)
df_ir['species'] =iris.target

df_ir.head(3)
```

![ent_11.png](./images/entropy/ent_11.png)


- 기준값과 컬럼명을 파라미터로 받아 조건부 엔트로피를 계산하는 함수

```python
def calc_cond_entropy(col, threshold) :
    df_ir['X1'] = df_ir[col] > threshold
    ptb = df_ir.groupby(['X1', 'species']).size().unstack().fillna(0)
    v = ptb.values
    pYX0 = v[0, :] / np.sum(v[0, :])
    pYX1 = v[1, :] / np.sum(v[1, :])
    HYX0 = sp.stats.entropy(pYX0, base=2)
    HYX1 = sp.stats.entropy(pYX1, base=2)
    HYX = np.sum(v, axis=1) @ [HYX0, HYX1] / np.sum(v)

    return HYX
```

- 컬럼명을 파라미터로 입력받아서 조건부 엔트로피 값을 반환받아 그래프를 그려주는 함수

```python
def plot_min_cond_entropy(col) :
    th_min = df_ir[col].min()
    th_max = df_ir[col].max()
    th_range = np.arange(th_min, th_max + 0.05, 0.05)

    cond_entropies = []
    for th in th_range :
        cond_entropies.append(calc_cond_entropy(col, th))

    id_min = np.argmin(cond_entropies)
    th_min = th_range[id_min]
    ce_min = np.min(cond_entropies)

    plt.figure(figsize=(8, 6))
    plt.plot(th_range, cond_entropies, 'r-')
    plt.plot(th_min, ce_min, 'bo')
    plt.title("{} 기준값에 따른 조건부엔트로피 ({:.3} 일 때 최소값 {:.3})" \
             .format(col, th_min, ce_min), y=1.03)
    plt.xlabel(col)
    plt.show()
```

#### 꽃받침의 길이를 0.05 간격으로 변화시켰을 때 조건부 엔트로피 값의 변화

```python
plot_min_cond_entropy('sepal length (cm)')
```
![ent_12.png](./images/entropy/ent_12.png)

#### 꽃받침의 폭을 0.05 간격으로 변화시켰을 때 조건부 엔트로피 값의 변화

```python
plot_min_cond_entropy('sepal width (cm)')
```

![ent_13.png](./images/entropy/ent_13.png)

- 베르시칼라와 버지니카 품종을 구분하기 위한 기준값은 꽃받침의 길이에서 찾는 것이 더 정확하다.
    - 조건부엔트로피 값이 낮기 떄문.


