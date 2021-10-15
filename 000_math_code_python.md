# python의 선형대수 코드

### 여러가지 행렬

- 전치연산 : A.T
- 영백터 : np.zeros([row, columns])
- 일벡터 : np.ones([row, columns])
- 정방행렬 :
    - np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
    - np.array(random.choices([random.randint(1, 10) for _ in range(20)], k=9)).reshape(3,3)
- 대각정방행렬 : np.diag([성분1, 성분2, 성분3, ...])
- 항등행렬 : np.identity(대각성분의수) : 모든 성분은 1
    - np.eye(대각성분의수)

### 행렬의 연산
- 연산 operator : x+y, x-y
- 내적 inner product : x.T @ y, np.dot(x.T, y) : 2차원 배열은 전치연산을 해줘야하고, 1차원 배열은 전치연산 자동인식
- 가중합 weighted sum : p.T @ n, np.dot(p.T, n)
- 제곱합 sum of squqred : x @ x.T
- 유사도 similarity : v1.T @ v2
- 행렬의 곱 : A @ B
    - A @ B != B @ A
    - (A+B)@C == A @ C + B @ C
    - (A+B).T == A.T + B.T
    - (A@B).T == B.T @ A.T
    - (A@B@C).T == C.T @ B.T @ A.T
- 행렬 X의 평균 : (np.ones([N,1]) @ {np.ones([N,1])}.T @ X) / N
- 항등행렬의 곱 : A @ I == I @ A == A

### 행렬의 크기
- 잔차제곱합 RSS residual sum of squares
    - 잔차벡터 : y - X @ w
    - 잔차제곱합 : e.T @ e
- 이차형식 quadratic form : x.T @ A @ x
    - 행렬의 부호를 판단하는 기준이 된다
    - 양의 정부호 : x.T @ A @ x > 0
    - 양의 준정부호 : x.T @ A @ x >= 0
- 행렬의 크기
    - 놈 : np.linalg.norm(A)
    - 대각합 : np.trace(A)
    - 행렬식 : np.linalg.det(A)
- 단위 벡터 unit vector : x * 1 / np.linalg.norm(x)
- 벡터의 선형조합 linear combination : x3 = 0.5 * x1 + 0.3 * x2

### 역행렬
- 역행렬 : np.linalg.inv(A)
- np.linalg.lstsq(A)
    - x : 최소자승문제의 답 x
    - resid : 잔차제곱합 RSS
    - rank : 랭크 : 선형독립인 벡터의 갯수
    - s : 특잇값 s
- 의사역행렬 : np.linalg.inv(A.T @ A) @ A.T

### 기하학과 선형대수
- 벡터의 투영성분 projection 과 직교성분 rejection
    - 투영성분의 길이 : a @ b / np.linalg.norm(b)
    - 투영성분 : projection = (a @ b / np.linalg.norm(b)) * (b / np.linalg.norm(b))
    - 직교성분 : a - projection

### 랭크
- 랭크 rank : np.linalg.matrix_rank(X1)
- 로우 랭크 low rank : 특잇값분해와 PCA 에서 사용
    - 랭크의 수는 M
    - 랭크-1 행렬 : np.linalg.matrix_rank(x @ x.T)
    - 랭크-2 행렬 : np.linalg.matrix_rank(x1 @ x1.T + x2 @ x2.T)
    - 랭크-M 행렬 : np.linalg.matrix_rank(x1 @ x1.T + x2 @ x2.T + x3 @ x3.T +...+)

### 좌표 변환
- 변환행렬 transform matrix
    - x = np.array([2,2])
    - g1 = np.array(1/np.sqrt(2), 1/np.sqrt(2))
    - g2 = np.array(-1/np.sqrt(2), 1/np.sqrt(2))
    - A = np.vstack([g1, g2]).T
- 새로운 기저벡터에 대한 좌표
    - 변환행렬의 역행렬 @ 원래 기저벡터에 대한 좌표
    - np.inv(A) @ x, np.inv(A).dot(x)

### 고유값 분해	
- 고유값 분해 eigenvalue-decomposition, eigen decomposition
    - A = np.array([[1, -2], [2, -3]]
    - w : 고유벡터, V : 고유벡터행렬
    - w, V = np.linalg.eig(A)
    - 수치계산으로 인한 소수점 오차 있음
    - Av = wv (w 고유값, v 고유벡터)
- 특성방정식 characteristic equation
    - 고유값을 구하는 공식
    - np.linalg.det(A-wI) = 0

### 대각화
- 대각화 diagonalization
    - 고유벡터행렬 @ 고유값행렬 @ 고유벡터행렬의 역행렬
    - np.linalg.eig(A) 에서 w는 벡터로 계산되므로, 대각행렬로 변환시켜줘야 한다.
    - V @ np.diag(w) @ np.inv(V)
- 대각화를 사용하여 A의 역행렬 계산
    - 고유벡터행렬 @ 고윳값 행렬의 역행렬 @ 고유벡터행렬의 역행렬
    - A^-1 = VW^-1V^-1
- 대칭행렬의 대각화
    - 고유벡터행렬 @ 고윳값행렬 @ 고유벡터행렬의 전치행렬
    - V @ np.diag(w) @ V.T

### 행렬의 랭크-1 형태 변환
- 대칭행렬의 랭크-1 행렬 표현
    - 고윳값과 랭크-1 행렬(고유벡터로 만듦)을 선형조합하면 원래 행렬 A가 된다.
    - 역행렬도 가능 : 고유값에 0이 없는 경우
    - w, V = np.linalg.eig(A)
    - v1 = V[:, 0:1]
    - v2 = V[:, 1:2]
    - v3 = V[:, 2:3]
    - w1, w2, w3 = w (고윳값의 수대로 변수에 각각 저장)
    - B1 = v1 @ v1.T, B2 = v2 @ v2.T, B3 = v3 @ v3.T
    - A = w1 @ B1 + w2 @ B2 + w3 @ B3

### 분산행렬
- 분산행렬
    - X.T @ X

### 특이값 분해
- 특잇값 분해 singularvalue-decomposition, singular decomposition
    - U : 왼쪽특이벡터행렬, 특잇값 대각행렬, 오른쪽특이벡터행렬
    - U, S, VT = np.linalg.svd(A)
    - VT 는 전치행렬로 출력된다.
    - A = U @ np.dig(S, 1)[:, 1:] @ VT
    - 특잇값행렬은 앞뒤의 행렬과 크기를 맞춰줘야하므로 대각행렬형태로 변환해야함.
- 특잇값 분해 축소형
    - 특잇값행렬의 원소가 0인 부분을 제외하고 앞뒤행렬에서 이와 대응하는 원소를 제외시켜준다.
    - U, S, VT = np.linalg.svd(A, full_matrices=False)

# PCA : 주성분 분석, 차원축소

### pca 모듈 임포트, 분석
```
from sklearn.decomposition import PCA
```

- 1차원으로 축소 : pca = PCA(n_components=1) 
- 1차원 근사데이터 집합 : X_low = pca.fit_transform(X) 
- 원래차원으로 복귀한 데이터 집합 : X2 = pca.inverse_transform(X_low) 

### 주성분 분석의 결과
- 데이터의 평균값 : pca.mean_
```
array([4.86, 3.31])
```

### 주성분 벡터, 단위기저 벡터
- pca.components_
```
array([[0.68305029, 0.73037134]])
```

### 주성분벡터는 원래행렬 - 평균값의 오른쪽특이벡터와 같다.
- X0 = X - X.mean(axis=0)
- U, S, VT = np.linalg.svd(X0)
- VT[:, 0]
```
array([-0.68305029, -0.73037134])
```

### 주성분벡터는 분산행렬의 고육벡터와 같다.
- Xcov = X0.T @ X0
- W, V = np.linalg.eig(Xcov)
- V[:, np.argmax(W)] : W 고윳값의 크기 순서대로 재배열 해줘야 순서가 맞다.
```
array([-0.68305029, -0.73037134])
```

### 차원축소한 값은 주성분 벡터와 원래 행렬(-평균)의 내적과 같다.
- x^_i = Wx_i
- X_low[7] == pca.components_ @ (X[7, :] - pca.mean_)

### 붓꽃데이터로 PCA
- pca 적용
```
X = iris.data
pca3 = PCA(n_components=1)
X_low = pca3.fit_transform(X)
X2 = pca3.inverse_transform(X_low)
```
- 결과 확인
- 평균값 확인 : pca3.mean_
```
array([5.84333333, 3.05733333, 3.758     , 1.19933333])
```
- 주성분 확인 : pca3.components_
```
array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ]])
```
- 특잇값행렬과 주성분 비교
```
X_minus_mean = X - X.mean(axis=0)
U, S, VT = np.linalg.svd(X_minus_mean)

VT[0, :]

array([ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ])
```

# 미분, 적분
- 슈와르츠 정리에 의해서 다변수함수의 편미분시 순서에 상관없이 결과가 같다.
- f_xy = f_yx

### 수치미분
- derivative의 인수로 미분할 함수와 해당지점 x 좌표, 이동거리 (dx) 를 입력한다.
- dx 는 작은 거리를 설정해주는데, 숫자가 너무 작아지면 오버플로우가 생겨 결과가 부정확해진다.
- 1e-6 : 백만분의 1
```
from scipy.misc import derivative

print(derivative(f, 0, dx=1e-6)

=====<print>=====

1.000000000001
```

### 심볼릭 연산 symbolic operator
- 미분을 손을 계산하듯이 변수들을 사용하여 나타내 준다.
- 계산 순서
	- 사용할 변수를 심볼로 정의한다.
	- 함수를 만든다.
	- 미분을 한다.
```
import sympy

# 라텍스 언어를 사용하기 위함
sympy.init_printing(use_latex='mathjax)
```
- 심볼을 정의할 때 'x y z' 식을 입력할 것
```
x, y = sympy.symbols('x y')
```
- 특수한 수학 지표들은 sympy.을 사용한다
```
f = sympy.exp(x**2 + 2 * y**2)
```
- 단변수, 다변수 함수의 미분, 편미분
```
sympy.diff(f, x)
sympy.diff(f, y)
sympy.diff(f, x, x)
sympy.diff(f, y, y)
sympy.diff(f, x, y)
sympy.diff(f, y, x)
```
- 미분결과 식을 정리하는 명령어
```
sympy.simplify(sympy.diff(f, x, y))
```

### 부정적분
- 미분의 반대 작업, 도함수를 통해서 원래 함수를 찾는 작업
```
import sympy

sympy.init_printing(use_latex='mathjax')
```
- 심볼 설정
```
x, y = sympy.symbols('x y')
```
- 함수 설정
```
f = (x*y)*sympy.exp(x**2 + y**2)
```
- 부정적분
```
sympy.integrate(f, x)
sympy.integrate(f, y)
sympy.simplify(sympy.integrate(f, x, x))
sympy.simplify(sympy.integrate(f, y, y))
sympy.simplify(sympy.integrate(f, x, y))
sympy.simplify(sympy.integrate(f, y, x))
```
