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
- 볼록도 convexity, 오목도 concave : 그래프 아래에서 위로 올려다 봤을 때의 형태를 떠올리면 쉽다.
    - 아래로 볼록한 그래프가 최소점, 위로 오목한 그래프가 최대점
- 볼록한 함수와 오목한 함수를 정의하는 식 : 볼록함수는 x에 t배 만큼 x2에 1-t배 만큼 곱한 값을 더한 위치의 함수값은 x, x2의 함수값에 각각 t, 1-t배 만큼 곱한 값의 합보다 항상 작다. 오목함수는 반대로 적용
    - 볼록함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20f%20%5Cleft%28%20t%7Bx%7D%20&plus;%20%281-t%29%7By%7D%20%5Cright%29%20%3C%3D%20tf%28%7Bx%7D%29%20&plus;%20%281-t%29%20f%28%7By%7D%29">
    - 오목함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20f%20%5Cleft%28%281-t%29%7Bx%7D%20&plus;%20t%7By%7D%20%5Cright%29%20%3E%3D%20%281-t%29f%28%7Bx%7D%29%20&plus;%20tf%7By%7D">
- 일반적으로 최적화는 컨벡스 문제를 다루며, 딥러닝도 복잡한 다변수 함수의 최적점을 찾는 문제와 같다. 
- 따라서 목적함수의 볼록도, 오목도에따라 최적화 방식도 다르다.

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
- 모든 입력변수로 편미분한 값이 0이어야 한다. (x_1 ~ x_N, 람다_1 ~ 람다_M, N+M 개의 입력변수) 

> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20x_1%7D%3D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_1%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20x_2%7D%3D%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_2%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cvdots">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_N%7D%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20%5Cdfrac%7B%5Cpartial%20g_j%7D%7B%5Cpartial%20x_N%7D%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20%5Clambda_1%7D%20%3D%20g_1%20%3D%200%2C">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cvdots">\
> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20%5Clambda_M%7D%20%3D%20g_M%20%3D%200">
    
- 이 연립방정식을 풀면 나오는 해에서 라그랑주 승수를 제외한 x_1~x_N 을 구할 수 있다.
- 최적화를 위한 2가지 조건
    - 기울기 필요조건 : 현재 지점 x가 최적점인지 판단하는 조건
        - 단변수함수 : x로 미분한 값 = 0
        - 다변수함수 : x1-xN 으로 편미분한 값이 모두 = 0
    - 다음 위치를 결정하는 알고리즘 : 최대경사법, 목적함수의 모양, 데이터의 특징에 따라서 여러가지 알고리즘이 있다.

### 라그랑주 승수의 의미
- 최적화 문제에서 제한조건이 있는 경우와 제한조건이 없는 경우의 최적화의 값이 다르려면 라그랑주 승수가 0이 아니어야한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clambda%20%5Cneq%200">
- 라그랑주 승수의 값이 0이면 제한조건이 있을때와 없을때의 최적화 값이 같게 된다.

### 부등식 제한조건이 있는 최적화 문제
- ``부등식 제한조건 inequality constraints`` : 등식 제한조건과 다르게 제한조건에 부등호가 있다.
    - 최적화 문제 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x%5E%7B*%7D%20%3D%20%5Carg%20%5Cmin_x%20f%28x%29">
    - 입력변수의 범위 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x%20%5Cin%20%5Cmathbf%20%7BR%7D%5EN">
    - 부등식 제한 조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%7Bg%7D_j%20%5Cleq%200%20%5C%3B%5C%3B%20%28j%3D1%2C%5Ccdots%2CM%29">
- 부등식 제한조건이 있는 최적화 문제도 라그랑주 승수법과 마찬가지로 목적함수를 h로 바꾸어서 풀 수 있다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%7Bh%7D%28x_1%2C%20x_2%2C%20%5Ccdots%2C%20x_N%2C%20%5Clambda_1%2C%20%5Clambda_2%2C%20%5Ccdots%2C%20%5Clambda_M%29">
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%7Bh%7D%28x%2C%20%5Clambda%29%20%3D%20f%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5EM%20%5Clambda_j%20g_j%28x%29">
- ``KKT Karush-Kuhn-Tucker 조건`` : 부등식 제한조건 최적화문제의 필요조건은 등식 제한조건 최적화문제와 다르다.
    - 모든 독립변수 x_1, x_2, ..., x_N 으로 미분한 값이 0이다. (등식제한조건에서는 모든 입력변수의 편미분값이 0이다.)
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Cdfrac%7B%5Cpartial%20h%28x%2C%20%5Clambda%29%7D%7B%5Cpartial%20x_i%7D%3D0">
    - 모든 라그랑주 승수와 제한조건 부등식을 라그랑주 승수로 미분한 값의 곱이 0이다.
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clambda_j%20%5Ccdot%20%5Cdfrac%7B%5Cpartial%20h%28x%2C%20%5Clambda%29%7D%7B%5Cpartial%20%5Clambda_j%7D%20%3D%200">
    - 라그랑주 승수는 음수가 아니어야 한다.
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clambda_j%20%5Cgeq%200">
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20g_j%20%3D%200%2C%20%5C%3B%5C%3B%20%5Clambda%20%5Cneq%200">
- ``부등식 제한조건이 있는 최적화 문제는 각 제한조건의 영역,범위에 대해서 KKT 조건을 따져보면서 최적해를 구해야 한다.``

#### KKT 2번째 조건의 의미
- 2번째 조건을 만족하는 경우는 라그랑주 승수 값 자체가 0이거나 미분한 값이 0이면 된다.
    - 즉 1번 조건과 달리 라그랑주 승수로 목적함수 h를 미분한 값은 0이거나 0이 아니어도 된다.
- 이 조건을 '라그랑주 승수의 의미'와 함께 생각해 보면
    - g_j가 0이면 λ는 0이 아니다. : 등식제한 조건이 된다. 그러므로 라그랑주 승수의 의미를 따른다. argminf와 argminh의 값이 다르다.
    - (g_j=0 자체가 등식제한 조건에 해당함)
    - g_j가 0이 아니면 λ는 0이다. : argminf와 argminh 값이 같으므로, 부등식제한 조건이 쓸모없다. 목적함수 h(x, λ)은 λg항이 있든 없든 항상 값이 같다.
- 그래프로 나타내면 부등식 제한조건이 허용하는 영역에 원래 목적함수 f의 최적해가 포함되어 있는지 아닌지 확인할 수 있다.
    - ``최적화 결과에 영향을 주지 않는 쓸모없는 제한조건`` : 제한조건이 있으나 없으나 원래 목적함수 f의 최적화 값과 같은 경우
        - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20g_j%20%5Cneq%200%2C%20%5C%3B%5C%3B%20%5Clambda%20%3D%200">
    - ``최적화 결과에 영향을 주는 등식인 제한조건`` : 제한조건에 의해 원래 목적함수 f의 최적화 값과 달라지는 경우
        - $g_j = 0, \;\; \lambda \neq 0$
- ``부등식 제한조건이 있는 최적화 문제는 각 제한조건의 영역,범위에 대해서 KKT 조건을 따져보면서 최적해를 구해야 한다.``

# 선형계획법 문제와 이차계획법 문제

### 선형계획법 문제
- ``선형계획법 linear programming`` : 방정식(등식제한조건)이나 부등식 제한조건을 가지는 선형 모형 linear model의 값을 최소화 하는 문제, LP문제
    > 선형 모형 : 선형 조합 linear combination 으로 만든 예측 모형
    > 선형 조합 : 벡터나 행렬에 스칼라를 곱한 후 더하거나 뺀 것
         >> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20c_1x_1&plus;c_2x_2&plus;%5Ccdots&plus;c_Nx_N"> \
         >> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20c_1A_1&plus;c_2A_2&plus;%5Ccdots&plus;c_NA_N"> 
         
    > 선형 회귀 모형 linear regression model : 독립변수 x에서 종속변수 y를 예측하는 방법, 단순하면서 널리쓰이는 예측모형
        >> <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Chat%20y%3Dw_1x_1%20&plus;%20w_2x_2%20&plus;%20%5Ccdots%20&plus;%20w_Nx_N%3Dw%5ETx">
- LP 문제의 목적함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Carg%20%5Cmin_x%20c%5ETx">
- 선형 연립방정식으로 된 등식 제한조건 eq_cons : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20Ax%3Db">
- 입력 변수 조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x%20%5Cgeq%200">
- LP 문제는 여러가지 형태가 있으며 이와 같은 형태를 선형계획법 문제의 기본형 standard from 이라고 한다.
    > 기본형을 확장한 정규형 canonical form : 등식 제한조건이 아닌 부등식 제한조건으로 바뀜\
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Carg%20%5Cmin_x%20c%5ETx%2C%20%5C%3B%5C%3B%20Ax%20%5Cleq%20b%2C%20%5C%3B%5C%3B%20x%20%5Cgeq%200">

### LP문제의 예시
- 어떤 공장에서 두 가지 상품을 생산하는 조건
    - 제품 A와 제품 B 각각 150개씩 생산해야 한다.
    - 생산 시간은 300시간 이다.
    - 제품 A는 생산하는데 1시간 걸리고, B는 2시간 걸린다.
    - 특정 부품이 8000개 밖에 없다.
    - A는 생산하는데 부품을 4개 필요하고, B는 6개 필요하다.
    - A의 생산이익은 3만원, B의 생산이익은 5만원이다.
    
- 제품 A와 B의 생산량을 x1, x2 라고 하면 A와 B를 각각 몇개씩 생산할 때 이익이 최대가 될까?
    > 목적함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20-3x_1%20-5x_1"> 
    
    > 부등식 제한조건 : \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20-x_1%20%5Cleq%20-150"> \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20-x_2%20%5Cleq%20-150"> \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x_1&plus;2%20x_2%20%5Cleq%20300"> \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%204%20x_1&plus;6%20x_2%20%5Cleq%208000"> 
       
    > 입력변수의 조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x_1%20%5Cgeq%200%2C%20%5C%3B%5C%3B%20x_2%20%5Cgeq%200">
       
- 정규형 선형계획법 문제로 표기
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Cmin%20%5Cbegin%7Bbmatrix%7D%203%20%26%205%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5Cend%7Bbmatrix%7D">
    
    >  <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Cmin%20%5Cbegin%7Bbmatrix%7D%20-1%20%26%200%20%5C%5C%200%20%26%20-1%20%5C%5C%201%20%26%202%20%5C%5C%204%20%26%206%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5Cend%7Bbmatrix%7D%20%5Cleq%20%5Cbegin%7Bbmatrix%7D%20-150%20%5C%5C%20-150%20%5C%5C%20300%20%5C%5C%208000%20%5Cend%7Bbmatrix%7D">
    
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5Cend%7Bbmatrix%7D%20%5Cgeq%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%200%20%5Cend%7Bbmatrix%7D">

### 이차계획법 문제
- ``이차계획법 문제 quadratic programming`` : 방정식(등식 제한조건)이나 부등식 제한조건을 가지는 일반화 된 이차형식 quadratic form의 값을 최소화하는 문제, QP 문제

- ``이차형식 quadratic form`` : 어떤 벡터와 정방행렬이 '행벡터x정방행렬x행벡터' 형식으로 되어 있는 것 
    > 선형회귀 분석의 결과로 얻은 예측값들과 이에 상응하는 실제값들 간의 차이를 잔차 residual라고 한다. (혹은 오차 erorr) \
    > 모든 입력변수에 대한 잔차를 구하고 이를 잔차벡터로 만들 수 있다.\
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20e%3Dy-Xw">  
        
    > 잔차의 크기 RSS Residual sum of square 는 잔차의 제곱합과 같다. \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Csmall%20%5Csum_%7Bi%3D1%7D%5EN%20%7Be_i%7D%5E2%3D%5Csum_%7Bi%3D1%7D%5EN%20%28y_i%20-%20w%5ETx_i%29%3De%5ETe%3D%28y-Xw%29%5ET%28y-Xw%29%3Dy%5ETy%20-%20y%5ETXw%20-%20w%5ETX%5ETy%20&plus;%20w%5ETX%5ETXw">  
    
    > 잔차의 크기인 잔차제곱합에서 도출 된 마지막 항, 벡터와 행렬의 곱의 형식 \
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Csmall%20w%5ETAw">
        
- QP의 목적함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Cdfrac%7B1%7D%7B2%7Dx%5ETQx%20&plus;%20c%5ETx"> 
- 등식 제한조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20Ax%20%3D%20b">
- 입력변수 조건 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20x%20%5Cgeq%200">
- 잔차 제곱합을 최소화하는 예측 모형에 추가적인 제한조건이 있으면 이차계획법 문제가 된다.
    - 이차계획법에 대해서 좀 더 알아 볼것!!!

# Python

### 1차원 목적함수의 최적점 확인
```python
def f1(x) :
    return (x-2)**2 + 2

xx = np.linspace(-1, 4, 100)
plt.plot(xx, f1(xx), 'g-')
plt.plot(2, 2, 'ro', ms=10)
plt.xlabel('$x$')
plt.ylabel('$f_1(x)$')
plt.title('1차원 목적함수')
plt.xlim(-1, 4)
plt.ylim(0, 10)

plt.show() ;
```











































