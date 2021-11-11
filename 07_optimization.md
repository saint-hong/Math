# 최적화

## 데이터분석의 목표
- 주어진 데이터로부터 우리가 원하는 출력데이터를 만들어주는 다양한 함수들 중에서 좋은 분석용 함수를 찾는다.
    - 함수의 성능을 고쳐서 더 좋은 함수로 만든는 방식으로 최적의 함수를 찾는다.
- 분석용 함수를 찾았다면 함수의 성능을 따져서 더 좋은 함수로 개선한다. 
- 데이터 분석의 목표 예측오차가 가장 작은 최적의 예측모형을 찾는 것이다. 
- 최적의 예측모형을 찾기위해서는 예측모형 함수의 모수를 변화시켜야 한다.
- 모수에 따라서 값이 달라지기 때문이다.
- 예측모형 함수를 목적함수 objective function 이라고 부른다.
- 목적함수의 종류에 따라서 함수의 최대값과 최소값의 의미가 달라진다.
    - 성능함수 performance function : 예측모형의 성능을 의미한다. 값이 클 수록 예측모형이 좋다.
        - 성능을 나타내는 값이므로 최대값의 위치가 중요하다.
    - 손실함수 loss function, 비용함수 cost function, 오차함수 erorr function : 예측값과 원래값의 오차, 차이를 나타낸다. 
        - 원래값과 예측값의 차이를 나타내므로 최소값의 위치가 중요하다.
- 즉 예측함수의 최대값 또는 최소값을 이루는 입력값 즉 모수를 찾는 것이 데이터분석의 목표이다.
    - 모든 데이터분석은 주어진 기준에 가장 적합한 수식을 찾는다는 의미
    - 따라서 일종의 최적화 optimization 문제를 푸는 과정이다.

# 최적화 기초

### 최적화 문제 
- 최적화 문제의 수식 : 함수 f(x)의 값을 max 또는 min 으로 만드는 x 의 값 x*
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x%5E%7B%5Cast%7D%20%3D%20%5Carg%20%5Cmax_x%20f%28x%29%24%20or%20%24x%5E%7B%5Cast%7D%20%3D%20%5Carg%20%5Cmin_x%20f%28x%29">
- 일반적으로 f(x) 를 -f(x) 로 바꾸어서 최소값을 찾는다. 
- 최소화하려는 함수 :
    - 목적함수 objective function, 비용함수, cost function, 손실함수 loss function, 비용함수 cost funtion, 오차함수 erorr function 등이 있다. 
    - J, C, L, E 로 표기 하는 경우가 많다.
- ``어떤 목적함수의 그래프에서 가장 작은 최소값에 해당하는 x 값을 구하는 것과 같다.``

### 그리드서치와 수치적 최적화
- ``그리드서치 grid search 방법`` : 일반적으로 어떤 목적함수에서 가장은 최소값을 찾는 방법은 여러가지 x값에 해당하는 출력값을 비교하는 것이다.
    - 단점 : x값에 해당하는 모든 목적함숫값을 구해야한다. 모든 트레이닝 데이터에 대해서 예측값과 타깃값의 차이를 구해야하므로 계산량이 매우 크다.
- ```수치적 최적화 numerical optimization ``` : 반복적 시행 착오 trial and erorr 에 의해서 최적화 필요조건에 만족하는 값 x* 을 찾는 방법이다. 
    - 함수의 위치가 최적점이 될때까지 가능한 최소의 횟수로 찾는 방식이다.
    - 두 가지 알고리즘을 필요로 한다. 
    >- 현재 위치 x_k가 최적점인지 확인하는 알고리즘
    >- 현재 위치가 최적점이 아닌경우 다음 위치로 옮기는 알고리즘

### 기울기 필요조건
- ``현재 위치가 최적점(최대 또는 최소)이라면 기울기와 도함수 값이 0이다.`` 
- 단변수 함수 : 미분값이 0 이어야 한다.

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Cdfrac%7Bdf%28x%29%7D%7Bdx%7D%20%3D%200">

- 다변수 함수 : 모든 입력변수에 대한 편미분값이 0 이어야 한다. 즉 그레디어트 벡터의 값이 0 이어야 한다.

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20f%28x_1%2C%20%5Ccdots%20%2C%20x_N%29%7D%7B%5Cpartial%20x_1%7D%3D0%20%2C%5C%3B%5C%3B%20%5Cdfrac%7B%5Cpartial%20f%28x_1%2C%20%5Ccdots%20%2C%20x_N%29%7D%7B%5Cpartial%20x_2%7D%3D0%2C%20%5C%3B%5C%3B%20%5Cdfrac%7B%5Cpartial%20f%28x_1%2C%20%5Ccdots%20%2C%20x_N%29%7D%7B%5Cpartial%20x_3%7D%3D0">

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Ccdots%2C%5C%3B%5C%3B%20%5Cdfrac%7B%5Cpartial%20f%28x_1%2C%20x_2%2C%20%5Ccdots%20%2C%20x_N%29%7D%7B%5Cpartial%20x_N%7D%3D0">

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Cnabla%20f%20%3D%200%2C%20g%20%3D%200">

- 기울기가 0이면 최대 또는 최소 지점이다. (반드시 최소지점은 아니다.)
    - 따라서 2차 도함수의 값을 구했을때 양수이면 최소값, 음수이면 최대값이다.
    
- 그레디언트 벡터 : 벡터 입력, 스칼라 출력 함수를 입력값 벡터로 미분한 것
    - 열벡터이고 모든 요소가 도함수로 이루어져 있다.
    - 함수를 한번 미분한 것
    - 다변수 함수의 그레디언트 벡터는 컨투어 플롯에서 기울기 방향을 나타내는 화살표와 같다. 

### 최대경사법
- ``최대경사법 steepest gradient decendent`` :  현재 위치에서의 기울기값을 이용하여 다음 위치를 찾는 방법
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x_%7Bk&plus;1%7D%20%3D%20x_k%20-%20%5Cmu%20%5Cnabla%20f%28x_k%29%20%3D%20x_k%20-%20%5Cmu%20g%28x_k%29">
    - 현재 지점 x_k에서 기울기값에 어떤 상수값을 곱한 것을 뺀다.
    - 현재 지점의 기울기가 음수이면 x_k보다 앞으로 이동하고, 기울기가 음수이면 x_k보다 뒤로 이동한다. 현재 지점의 기울기는 다음 위치의 방향을 나타낸다.
    - 비례상수 mu :  ``스텝 사이즈 step size``, 이동 거리의 비율을 정한다.
    - 스텝 사이즈는 사용자가 경험적으로 정하거나 알고리즘으로 구한다. `적절한 스텝사이즈를 구해야 한다`
- 최대경사법에서는 스텝사이즈와 시작점에 따라서 최적화 결과가 결정된다.
    - 스텝사이즈가 너무 크면 최적점으로부터 멀어지게 된다.
    - 다변수 함수의 컨투어 플롯에서는 곡면이 계곡과 같은 지점에서 시작할 경우 진동 현상 oscillation 이 발생한다. 좌우를 왔다갔다하면서 최적점을 찾는데 오래 걸린다. 
- 진동 현상을 없애는 방법
    - 헤시안 행렬 (2차 도함수) 를 이용하는 방법
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20Hf%28x%29%20%3D%20J%28%5Cnabla%20f%28x%29%29%5ET%24%2C%20%24H_%7Bi%2Cj%7D%20%3D%20%5Cdfrac%20%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_i%20%5Cpartial%20x_j%7D">
        - 그레디언트 벡터를 자코비안 행렬로 만든 뒤 전치연산을 한 것
        - 정방행렬, 대칭행렬, 
    - 모멘텀 momentom 방법 : x_k+1 의 위치를 일정한 방향을 유지하도록 모멘텀 성분을 추가하는 방법, 인공신경망 등에서 쓰임

### 2차 도함수를 사용하는 뉴턴 방법
- ``뉴턴 방법 Newton method`` : 목적함수가 2차함수 일 경우 최적점을 한번에 찾아준다.
- 한번에 찾아준다는 점이 특징이다. 
- 단변수 함수일 경우 : 현재 지점과 그 주변의 지점에서의 1차 도함수와 2차 도함수를 이용하여 최적점을 찾는방식
    - 테일러 전개식을 사용하여 현재 지점과 주변의 어떤 지점 t를 나타낸다.
    - 다음식에 의해서 어떠한 점에서 시작하더라도 한번에 최적점을 찾는다.

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20f%28x_k%20&plus;%20t%29%20%5Capprox%20f%28x_k%29%20&plus;%20f%5E%7B%27%7D%28x_k%29t%20&plus;%20%5Cdfrac%20%7B1%7D%7B2%7D%20f%5E%7B%27%27%7D%28x_k%29t%5E2"> \
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Cdfrac%20%7Bd%7D%7Bdt%7D%28f%28x_k%29%20&plus;%20f%5E%7B%27%7D%28x_k%29t%20&plus;%20%5Cdfrac%7B1%7D%7B2%7Df%5E%7B%27%27%7D%28x_k%29t%5E2%29%20%3D%200"> \
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20f%5E%7B%27%7D%28x_k%29%20&plus;%20f%5E%7B%27%27%7D%28x_k%29t%20%3D%200"> \
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20t%3D-%5Cdfrac%20%7Bf%5E%7B%27%7D%28x_k%29%7D%7Bf%5E%7B%27%27%7D%28x_k%29%7D"> \
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x_%7Bk&plus;1%7D%20%3D%20x_k%20&plus;%20t%20%3D%20x_k%20-%20%5Cdfrac%20%7Bf%5E%7B%27%7D%28x_k%29%7D%7Bf%5E%7B%27%27%7D%28x_k%29%7D">
    
- 다변수 함수일 경우 : 최대경사법에서 mu 대신에 2차 도함수인 헤시안 행렬의 역행렬을 곱한다.
    - ``다음 식에 의해서 어떤 점 x_n 에서 시작하더라도 바로 최저점으로 이동한다.``

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%7Bx%7D_%7Bn&plus;1%7D%20%3D%20%7Bx%7D_n%20-%20%5B%7BH%7Df%28%7Bx%7D_n%29%5D%5E%7B-1%7D%20%5Cnabla%20f%28%7Bx%7D_n%29"> \
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%7Bx%7D_%7Bn&plus;1%7D%20%3D%20x_n%20-%20%5Cdfrac%20%7Bf%5E%7B%27%7D%28x_n%29%7D%7Bf%5E%7B%27%27%7D%28x_n%29%7D"> 
    
- 뉴턴방법은 도함수 (그레디언트 벡터)와 2차 도함수 (헤시안 행렬)를 둘다 구해야하고 2차 함수의 형태가 아닌 경우 최적화가 안되는 경우가 있다.

### 준뉴턴 방법
- ``준뉴턴 방법 Quasi-Newton method`` : 뉴턴 방법을 기반으로 단점을 보완한 방법, 헤시안 행렬 대신에 기울기 벡터를 분석하고 업데이트하여 최적점을 찾는다. 
    - 준뉴턴 방법으로는 BFGS, SR1 fomula, BHHH 방법 등이 있으며, BFGS 방법이 일반적으로 많이 사용 된다.
    - ``BFGS 방법 Broyden-Fletcher-Goldfarb-shanno`` : 자코비안 행렬을 업데이트하여 최적점을 찾는다.
    - ``CG 방법 conjugated gradient`` : 헤시안 행렬 대신 변형된 그레디언트 벡터를 바로 계산한다. 

### 여러가지 최적화 방법들
- 이 외에도 함수의 종류나 데이터의 형태 등에 따라서 다양한 최적화 방법이 있다.
- 핵심은 기울기 필요조건을 만족하면서 어떤 함수냐에 따라 다음 지점을 찾는 효율적인 방법을 찾아내는 것이다.
- scipy.github 페이지에 여러가지 최적화 방법들이 소개되어 있다.
    - scalar functions optimization
        - minimize_scalar(method='') : brant, bounded, golden
    - local (multivariate) optimization
        - minimize(method='') : CG, BFGS, Newton-CG, Powell, TNC, L-BFGS-B, SLSQP, ...
    - global (multivariate) optimization
        - basinhopping, brute, differntial_evolution, ...

### 전역 최적화 문제
- 최적화하려는 함수에 복수의 국소 최저점 local minima 있을 경우 수치적 최적화 방법으로 전역 최적점 global minimum에 도달하지 못할 수도 있다.
    - 전역 최적점 : 전체 함수에서 가장 작은 지점
    - 국소 최적점 : 함수의 특정 구간에서 가장 작은 지점
    - 전역 최적점이 국소 최적점보다 더 작다.
    - 국소 최적점은 한개일 수도 있고, 여러개일 수도 있다.
- 초기 추정값이나 알고리즘에 따라서 결과가 달라지게 된다. 
- 여러가지 다른 최적화 알고리즘을 사용해 보면 좋을 것 같다.

### 컨벡스 문제
- 컨벡스 문제 convex problem : 목적함수의 2차 도함수 값이 항상 0 이사이 되는 영역에서만 정의된 최적화 문제
- 2차 도함수 값이 0 이상이면 원래 목적함수의 형태는 볼록한 형태 (볼록도 convex) 이므로 최소점을 갖는 함수이다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x%5E2%7D%3D0">
- 다변수 목적함수에서는 주어진 영역ㅇ서 헤시안 행렬은 항상 양의 준정부호 positive semidefinite 이다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x%5ETHx%20%3E%3D%200%20%5C%3B%5C%3B%20for%20%5C%3B%20all%20%5C%3B%20x">
    
# 제한조건이 있는 최적화 문제
- 제한조건이 있는 최적화 constrained optimization

### 등식제한 조건이 있는 최적화 문제
- 등식제한 조건 equality constraint
- 최적화의 조건과 등식제한 조건을 함께 충족해야한다. : M개의 연립방정식이 0이어야 한다.
    - 목적함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x%5E%7B%5Cast%7D%20%3D%20%5Carg%20%5Cmin_%7Bx%7D%20f%28x%29">
    - 입력변수의 크기 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20x%20%5Cin%20%5Cmathbf%20%7BR%7D%5EN">
    - 등식제한 조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20g_j%28x%29%3D0%20%5C%3B%5C%3B%20%28j%3D1%2C%5Cldots%2CM%29">
    
### 라그랑주 승수법
- 라그랑주 승수법 lagrange multiplier : 등식제한 조건이 있는 최적화 문제를 푸는 방법, 라그랑주 승수 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Clarge%20%5Clambda"> 를 사용한다.
- 새로운 목적함수 h(x, 람다)를 만든다. 원래 목적함수 f(x) 와 라그랑주 승수인 새로운 변수를 등식제한 조건 g_j 와 곱한 후 더한다. 
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20h%28x%2C%20%5Clambda%29%3D%20h%28x_1%2C%20x_2%2C%20%5Cldots%2C%20x_N%2C%20%5Clambda_1%2C%20%5Clambda_2%2C%20%5Cldots%2C%20%5Clambda_M%29%20%7B%5Ccolor%7BGreen%7D%20%7B%5Ccolor%7BBlue%7D%20%7D%7D%3D%20f%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Clambda_j%20g_j%28x%29">
- 새로운 함수 h는 라그랑주 승수 M개를 추가했기때문에 입력변수는 x N개, 람다 M개 이다.
- 입력변수가 늘어난 만큼 그레디언트 벡터를 영벡터로 만드는 최적화 조건이 된다.
- 모든 입력변수로 편미분한 값이 0이어야 한다. 

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20x_1%7D%3D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_1%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20x_2%7D%3D%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_2%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Ccdots%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_N%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_N%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20%5Clambda_1%7D%20%3D%20g_1%20%3D%200%2C">\ 
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Ccdots%2C">\  
><img src="https://latex.codecogs.com/gif.latex%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20%5Clambda_M%7D%20%3D%20g_M%20%3D%200">

> $\begin{aligned}
    \dfrac{\partial h}{\partial x_1}
    &= \dfrac{\partial f}{\partial x_1} + \sum_{j=1}^{M} \dfrac{\partial g_j}{\partial x_1} = 0 \\
    &= \dfrac{\partial f}{\partial x_2} + \sum_{j=1}^{M} \dfrac{\partial g_j}{\partial x_2} = 0 \\
    &\vdots \\
    &= \dfrac{\partial f}{\partial x_N} + \sum_{j=1}^{M} \dfrac{\partial g_j}{\partial x_N} = 0 \\
    &= \dfrac{\partial h}{\partial \lambda_1} = g_1 = 0 \\
    &\vdots \\
    &= \dfrac{\partial h}{\partial \lambda_M} = g_M = 0 \\
    \end{aligned}$
    
- 이 연립방정식을 풀면 나오는 해에서 라그랑주 승수를 제외한 x_1~x_N 을 구할 수 있다.

