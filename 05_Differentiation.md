# 데이터분석과 미분

### 1. 데이터분석에서의 미분의 의미

#### `예측모형의 성능과 미분`
- `데이터분석의 목표는 어떤 데이터의 예측을 위한 최적의 예측 모형을 찾는 것이다.` 최적의 예측 모형은 모수를 잘 선택하여 모형의 성능을 높이는 , 즉 최적화하는 과정과 같다.
- 예를들어 어떤 데이터에 대한 선형예측모형의 예측값은 가중치 w와 데이터의 변수 x의 선형조합으로 만들어진다.
> <img src"https://latex.codecogs.com/gif.latex?%5Chat%20y%20%3D%20w_1x_1%20&plus;%20w_2x_2%20&plus;%20%5Ccdots%20&plus;%20w_Nx_N"/>
- 이 때에 가중치 w가 모수(=계수)가 되며 **모수를 어떤 것을 선택하느냐에 따라서 모형의 성능이 달라지게 된다.** 모형의 성능은 크기를 비교하기 위한 값이므로 스칼라가 되어야 한다.
- 모수 선택에서 모형의 성능 측정까지의 과정은 다변수함수를 계산하는 과정과 같다. 이렇듯 예측모형의 성능을 높이기 위한 함수를 **성능함수 performance function** 라고 한다. 성능함수의 값은 클 수록 좋다.
- 반대로 모형의 성능값이 아닌 모형의 예측값과 목표값을 비교하여 얼마나 차이가 있는지 오류를 따져 이 오류를 작게하는 함수로 **손실함수 loss function, 비용함수 cost function, 오류함수 error functiion** 이 있다.
- 이렇듯 최적화의 대상이 되는 함수들, 성능함수, 손실함수, 비용함수, 오류함수를 **목적함수**라고 한다. 목적함수를 선택한 후에는 모수를 조절함으로써 목적함수를 크게 또는 작게할 수 있다.
- 목적함수는 모수 w 를 입력받아 성능 또는 손실 값을 출력해주며, 입력변수 인 w를 조정하면 출력값도 변하게 된다. 입력변수의 변화에 따라 출력값의 변화하는 비율을 보여주는 것을 **미분 diffrentiation**이라고 한다.
- `즉 데이터 분석에서의 미분의 의미는`, 최적의 예측모형을 판단하는 기준으로 모형의 성능을 높이기위하여 적절한 목적함수를 가지고 입력변수로 가중치 w를 받아 조절함으로써 이와 함께 변하는 성능(손실)인 출력값과의 변화 비율을 나타내는 **일종의 신호**와 같다.
> 분석할 데이터 -> 목적함수 선택 -> 입력변수 가중치 w 조절 -> 변화하는 출력값 -> 미분 적용 -> 모수 w의 변화와 성능(손실) 출력값의 변화에 대한 변화율 확인 -> 모형의 성능 확인 -> 성능이 좋은 것 또는 오류가 작은 모형 선택 -> 최적의 예측 모형 도출

- 이러한 과정을 **최적화 optimization** 이라고 한다.

#### `PCA principle component analysis 에서의 최적화와 미분 `
- 주성분 분석 즉 PCA 의 최적화 과정을 떠올려보면, 원래 행렬과 역변환 행렬을 사용하여 만든 행렬의 차를 가장 작게 하는 문제가 되었다.
- <img src="https://latex.codecogs.com/gif.latex?arg%5C%3Bmin_%7B%5Chat%20x%7D%20%7B%5CVert%20x_i%20-%20U%5Chat%20x%5CVert%7D%5E2"/>
- 최적화의 대상이 된 목적함수를 최소화하기 위한 **차원축소 벡터 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20x"/>** 를 찾는 문제였다. **이를 위해 목적함수를 미분하여 얻은 값**이 영벡터가 되는 값 즉 역변환 행렬과 변환 행렬의 관계를 푸는 문제가 되었다.
    > - <img src="https://latex.codecogs.com/gif.latex?-2U%5ETx%20&plus;%202%5Chat%20x%20%3D%200%20%5C%3B%20%5Crightarrow%20%5C%3B%20%5Chat%20x%20%3D%20U%5ETx"/>
    >> <img src="https://latex.codecogs.com/gif.latex?%5Chat%20x%20%3D%20Wx"/>
    > - <img src="https://latex.codecogs.com/gif.latex?U%3DW%5ET"/>
- 처음의 최적화 식에 위의 값을 대입하고 모든 벡터에 대해 적용하면, 최적의 변환 행렬 W 를 찾는 문제인 랭크-k 근사 문제가 되며, W 는 가장 큰 k 번쨰까지의 특잇값에 해당하는 오른쪽특이벡터임을 알 수 있었다.
    - <img src="https://latex.codecogs.com/gif.latex?arg%5C%3Bmin_%7BW%7D%5C%3B%7B%5CVert%20X%20-%20XW%5ETW%20%5CVert%7D%5E2"/>
- 즉 PCA 과정에서 사용된 목적함수와 미분은, 모수의 변화에 대한 예측모형의 성능값의 변화율을 확인하여 최적의 모형을 만드는 모수 W를 찾는 역할을 했다는 것을 알 수 있다.

### 2. 미분

#### `기울기 slope`
- `기울기` : 어떤 함수관계에서의 입력변수 x의 변화에 따른 출력변수 y값의 변화율 **정보**, 민감도 sensitivity 라고도 한다.
    - 고장난 라디오의 음량 조절나사의 각도를 x라고 할때 출력되는 음량값을 y라고 할 수 있다. 일상적으로 조절나사를 좌우로 잘 돌리다보면, 어떤 지점에서 소리가 커지는지 경험적으로 파악할 수 있다.
    - `음량이 최대가 되는 조절나사의 각도를 찾는 문제는 변수 x에 대한 최적화문제라고 할 수 있다.`
- `수치적 최적화`는 가장 적은 방법을 시도하여 가장 큰 음량일 때의 최적의 x를 찾는 것을 의미한다. 가장 적은 횟수로 최적의 x를 찾으려면 다음과 같은 정보를 따라야한다.
    - x1의 위치에서 각도를 증가시켰을 때 음량 y가 커졌다면, 다음 시도인 x2는 x1보다 큰 값이어야 가장 큰 y값에 다가간다.
    - x1의 위치에서 각도를 증가시켰을 때 음량 y가 작아졌다면, 다음 시도인 x2는 x1보다 작은 값이어야 가장 큰 y값에 다가간다.
- 즉 기울기는 음량이 가장 큰 y값을 찾기 위해 조절나사의 각도 x를 x1, x2, x3, ... 최소한의 횟수로 변화시키려고 할 때, x1의 다음값을 어떻게 설정하면 좋을지에 대한 정보를 제공해준다.

#### `그래프에서의 기울기`
- 입력변수 x의 변화(x2-x)에 대한 출력변수의 변화(f(x2)-f(x))의 비율
    > <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%20%7B%5CDelta%20y%7D%7B%5CDelta%20x%7D%20%3D%20%5Cdfrac%20%7Bf%28x_2%29-f%28x%29%7D%7Bx_2-x%7D%3D%5Cdfrac%7Bf%28x&plus;%5CDelta%20x%29-f%28x%29%7D%7B%5CDelta%20x%7D"/>
- 그래프에서의 기울기는 x의 변화량인 <img src="https://latex.codecogs.com/gif.latex?%5CDelta%20x"/>가 0으로 근접해 갈 때의 변화율이다.
- 한 점에서의 접선의 기울기는 x의 변화량에 따라 달라질 수 있다. x2, x3, x4 과의 tangent 값이 달라지기 때문이다. **따라서 기울기를 통일해 주기 위해, x의 변화량을 무한대로 0에 가깝게 만드는 방식을 사용.**
    > <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20slope%20%3D%20%5Clim_%7B%5CDelta%20x%20%5Crightarrow%200%7D%20%5Cdfrac%7Bf%28x&plus;%5CDelta%20x%29-f%28x%29%7D%7B%5CDelta%20x%7D"/>

#### `수치미분`
- 기울기를 정확하게 구할 수 있는 방법
- scipy.misc 패키지의 derivative() 명령어를 사용하면 기울기를 구할 수 있다. 정확한 값은 아니다.
- 인수로 함수 f, 좌표 x, 이동할 거리 dx 가 사용된다. dx는 작을 수록 좋지만, 너무 작으면 부동소수점 연산의 오버플로우 오류가 발생하여 역으로 오차가 증폭할 수 있으므로 주의해야한다.
    - `부동소수점 연산의 오버플로우` : 정수는 이진법으로 표현 가능하지만, 소수는 어렵다. 소수는 원래 무한대의 수이다. 파이썬은 소수를 17 자리까지만 나타내준다. 모든 수는 이러한 오차가 있다고 봐야한다.

#### `미분 differentiation`
- 미분 : 어떤 함수로부터 그 함수의 기울기를 출력하는 새로운 함수를 만드는 작업이다. 미분은 동사이다.
    - 새로운 함수 = 기울기 출력 함수
- 미분의 표기법
> <img src="https://latex.codecogs.com/gif.latex?f%27%3D%5Cdfrac%7Bd%7D%7Bdx%7D%28f%29%3D%5Cdfrac%7Bd%7D%7Bdx%7Df%3D%5Cdfrac%7Bdf%7D%7Bdx%7D%3D%5Cdfrac%7Bd%7D%7Bdx%7D%28y%29%3D%5Cdfrac%7Bd%7D%7Bdx%7Dy%3D%5Cdfrac%7Bdy%7D%7Bdx%7D"/>

#### `미분 가능`
- 모든 함수가 미분 가능한 것은 아니다. 어떤 경우는 미분을 할 수 없는 경우도 있다.
- 미분을 할 수 없는 경우는 **미분 불가능** 이라고 하고, 미분이 가능한 경우는 **미분 가능** 이라고 한다.
    - ReLU 함수에서 x=0 인 경우는 미분 불가능에 해당한다.
    - x >0 이면 기울기는 0 이지만, x < 0 이면 기울기가 1 이다. x = 0 인 경우는 기울기를 구할 수 없다.

#### `미분공식`
- 몇 가지 미분공식을 조합하여 복잡한 함수의 도함수를 구할 수 있다.
    > - 기본미분공식
    >> 상수미분, 거듭제곱미분, 지수미분, 로그미분
    > - 선형조합법칙, 곱셈법칙, 연쇄법칙

- `기본미분공식`
    - 상수미분 : 상수를 미분하면 0 이 된다. (기울기 0의 의미) <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28c%29%3D0"/>
    - 거듭제곱미분 : n 제곱이 n-1 제곱이 되고, n 은 지수와 곱해진다. <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28x%5En%29%3Dnx%5E%7Bn-1%7D%24%2C%20%24%5Cfrac%7Bd%7D%7Bdx%7D%28x%5E%7B-2%7D%29%3D-%5Cfrac%7B2%7D%7Bx%5E3%7D"/>
    - 로그미분 : 로그를 미분하면 지수가 역수로 변환된다. <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28%5Clog%7Bx%7D%29%3D%5Cfrac%7B1%7D%7Bx%7D"/>
    - 지수미분 : 밑이 오일러인 지수를 미분해도 변하지 않는다. <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28e%5Ex%29%3De%5Ex"/>
- `선형조합법칙`
    - 어떤 함수에 상수를 곱한 함수를 미분하면, 도함수에 상수를 곱한 것과 같다. <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28c%20%5Ccdot%20f%29%3Dc%20%5Ccdot%20%5Cfrac%7Bdf%7D%7Bdx%7D"/>
    - 두 함수를 더한 함수를 미분하면, 각 도함수를 합한 것과 같다. <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdx%7D%28f_1&plus;f_2%29%3D%5Cfrac%7Bdf_1%7D%7Bdx%7D&plus;%5Cfrac%7Bdf_2%7D%7Bdx%7D"/>
- `곱셈법칙`
    - 두 함수를 곱한 함수의 미분은 각 개별 함수의 도함수를 사용하여 원래 함수의 도함수를 구한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%28f%20%5Ccdot%20g%29%3Df%5Ccdot%5Cdfrac%7Bdg%7D%7Bdx%7D%20&plus;%20g%5Ccdot%5Cdfrac%7Bdf%7D%7Bdx%7D"/>
- `연쇄법칙 chain rule`
- 미분하고자 하는 함수의 입력변수가 다른함수의 출력변수인 경우에 적용할 수 있다.
    - <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dh%28g%28x%29%29%5C%3B%5Crightarrow%5C%3B%20%5Cdfrac%7Bdf%7D%7Bdx%7D%3D%5Cdfrac%7Bdh%7D%7Bdg%7D%20%5Ccdot%20%5Cdfrac%7Bdg%7D%7Bdx%7D"/>
    - 함수가 복잡할 경우에는 f, h, g, y 등의 중간변수를 만들고 함수의 관계에 따라서 구분지은 후 연쇄법칙을 적용한다.
        > - <img src="https://latex.codecogs.com/gif.latex?f%3D%5Cexp%5Cdfrac%7B%28x-%5Cmu%29%5E2%7D%7B%5Csigma%5E2%7D"/>
        >> <img src="https://latex.codecogs.com/gif.latex?f%3D%5Cexp%28z%29%2C%5C%3Bz%3D%5Cdfrac%7By%5E2%7D%7B%5Csigma%5E2%7D%2C%5C%3By%3Dx-%5Cmu"/> \
        >> <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bdf%7D%7Bdx%7D%3D%5Cdfrac%7Bdf%7D%7Bdz%7D%5Ccdot%5Cdfrac%7Bdz%7D%7Bdy%7D%5Ccdot%5Cdfrac%7Bdy%7D%7Bdx%7D"/> \
        >> <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bdf%7D%7Bdz%7D%3D%5Cexp%28z%29%3D%5Cexp%5Cdfrac%7B%28x-%5Cmu%29%5E2%7D%7B%5Csigma%5E2%7D%24%2C%20%24%5Cdfrac%7Bdz%7D%7Bdy%7D%3D%5Cdfrac%7B2y%7D%7B%5Csigma%5E2%7D%3D%5Cdfrac%7B2%28x-%5Cmu%29%7D%7B%5Csigma%5E2%7D%24%2C%20%24%5Cdfrac%7Bdy%7D%7Bdx%7D%3D1"/>
        > - <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bdf%7D%7Bdx%7D%20%3D%20%5Cdfrac%7B2%28x-%5Cmu%29%7D%7B%5Csigma%5E2%7D%5Ccdot%5Cexp%5Cdfrac%7B%28x-%5Cmu%29%5E2%7D%7B%5Csigma%5E2%7D"/>

#### `2차 도함수 second derivative`
- `도함수의 기울기`
- 도함수를 미분하여 만든 도함수를 말한다. 함수 f의 기울기 출력함수인 도함수를 다시 미분하여 도함수의 기울기를 출력해주는 함수이다.
    - <img src="https://latex.codecogs.com/gif.latex?f%5E%7B%5Cprime%5Cprime%7D%28x%29%3D%5Cdfrac%7Bd%5E2f%7D%7Bdx%5E2%7D%3D%5Cdfrac%7Bd%5E2y%7D%7Bd%5E2x%7D"/>
- 함수, 도함수, 2차 도함수의 관계 (오목과 볼록의 기준은 아래에서 올려다본 시점, 오목은 봉우리, 볼록은 계곡)
    - 함수가 **오목** concave -> 도함수값 **감소** -> 2차 도함수값 **음수**
    - 함수가 **볼록** convex -> 도함수값 **증가** -> 2차 도함수값 **양수**
    - `2차 도함수값을 볼록도 convexity 라고도 부른다.`

#### `편미분 partial differentiation`
- 다변수 함수의 미분
- 다변수이기때문에 변수 각각에 대해서 따로 미분이 가능하다. 따라서 하나의 함수에서 여러개의 도함수가 나올 수 있다.
- 어떤 하나의 독립변수에 대해 미분을 할 때는 다른 독립변수를 상수 처럼 취급하여 계산한다.
    > <img src="https://latex.codecogs.com/gif.latex?f_%7Bx%7D%28x%2Cy%29%20%3D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D"/>,   <img src="https://latex.codecogs.com/gif.latex?f_%7By%7D%28x%2Cy%29%20%3D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D"/>

#### `다변수함수의 연쇄법칙`
- 다변수함수가 연결 되어 있을때에도 연결법칙을 사용하여 미분할 수 있다.
- N개의 함수 f1, f2, ..., fN 가 입력변수 x를 가질때 출력변수 y1, y2, ..., yN 라고 한다. yN 을 입력받는 함수 g와 출력 z라고 한다. 이때 변수 x값의 변환에 따른 z의 값의 변화
    >- <img src="https://latex.codecogs.com/gif.latex?y_1%20%3D%20f_1%28x%29%2C%5C%3B%5C%3B%20y_2%3Df_2%28x%29%2C%5C%3B%5C%3B%20y_N%3Df_N%28x%29"/>
    >- <img src="https://latex.codecogs.com/gif.latex?z%20%3D%20g%28y_1%2C%20y_2%2C%20%5Ccdots%2C%20y_N%29"/>
    >- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bdz%7D%7Bdx%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_1%7D%5Cdfrac%7Bdy_1%7D%7Bdx%7D%20&plus;%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_2%7D%5Cdfrac%7Bdy_2%7D%7Bdx%7D%20&plus;%20%5Ccdots%20&plus;%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_N%7D%5Cdfrac%7Bdy_N%7D%7Bdx%7D"/>
    >- 편미분*미분 + 편미분*미분 + ... 의 형태이다.
- N개의 함수 f1, f2, ..., fN이 x1, x2, ..., xM 개의 입력변수를 갖는 다변수함수라고 할 때, 변수 x1값의 변화에 따른 z의 변화
    >- <img src="https://latex.codecogs.com/gif.latex?y_1%20%3D%20f_1%28x_1%2Cx_2%2C%5Ccdots%2Cx_M%29%2C%5C%3B%5C%3B%20y_2%3Df_2%28x_1%2Cx_2%2C%5Ccdots%2Cx_M%29%2C%5C%3B%5C%3B%20y_N%3Df_N%28x_1%2Cx_2%2C%5Ccdots%2Cx_M%29"/>
    >- <img src="https://latex.codecogs.com/gif.latex?z%20%3D%20g%28y_1%2C%20y_2%2C%20%5Ccdots%2C%20y_N%29"/>
    >- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x_1%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_1%7D%5Cdfrac%7B%5Cpartial%20y_1%7D%7B%5Cpartial%20x_1%7D%20&plus;%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_2%7D%5Cdfrac%7B%5Cpartial%20y_2%7D%7B%5Cpartial%20x_1%7D%20&plus;%20%5Ccdots%20&plus;%20%5Cdfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y_N%7D%5Cdfrac%7B%5Cpartial%20y_N%7D%7B%5Cpartial%20x_1%7D"/>
    >- 편미분*편미분 + 편미분*편미분 + ... 의 형태이다. 변수 x2, x3, ..., xM 에 대해서도 똑같이 적용된다.

#### `2차 편미분`
- 편미분에 대한 2차 도함수. 즉 다변수 함수의 도함수의 도함수.
- 2차 편미분 방법은 1차 편미분을 하고, 그 결과를 가지고 다시 편미분을 한다. 이 과정에서 여러개의 변수 중 어떤 것을 선택할지는 자유롭게 정할 수 있다.
- 순서데로 선택한 변수를 아래첨자를 써서 표기한다.
    >- <img src="https://latex.codecogs.com/gif.latex?f_%7Bxx%7D%28x%2Cy%29%3D%5Cdfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial%20x%5E2%7D"/>
    >- <img src="https://latex.codecogs.com/gif.latex?f_%7Byy%7D%28x%2Cy%29%3D%5Cdfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial%20y%5E2%7D"/>
    >- <img src="https://latex.codecogs.com/gif.latex?f_%7Bxy%7D%28x%2Cy%29%3D%5Cdfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial%20y%20%5Cpartial%20x%7D"/>   (순서 바뀜)
    >- <img src="https://latex.codecogs.com/gif.latex?f_%7Byx%7D%28x%2Cy%29%3D%5Cdfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial%20x%20%5Cpartial%20y%7D"/>   (순서 바뀜)
- `슈와르츠 정리`
- 연속 함수이고, 미분 가능한 함수인 경우 미분의 순서가 바뀌어도 결과는 같다.
    - <img src="https://latex.codecogs.com/gif.latex?f_%7Bxy%7D%20%3D%20f_%7Byx%7D"/>
- 변수가 여러개이고 동시에 함수들이 서로 연결되어 있는 경우는 1차, 2차 편미분 과정에서 연쇄법칙과 곱셈법칙이 여러번 사용될 수 있다.

#### `심파이 SymPy`
- 심파이는 심볼릭연산 symbolic operation 을 지원하는 파이썬의 패키지이다. 심볼릭 연산은 사람이 손으로 미분을 계산하는 것과 같은 방식으로 연산하는 기능을 말한다.
- 일반적으로 파이썬에서 연산을 하려면 변수 x 에 값을 저장하여 변수선언을 해주어야 한다. 그러나 심볼릭 연산은 변수선언 없이도 x 를 심볼화하여 숫자계산 뿐만 복잡한 미분과 적분의 연산도 처리해준다.
- 딥러닝 deep learning 등에서 사용되는 텐서플로 패키지나 파이토치 패키지에서도 심볼릭 연산을 지원한다.
- 데이터 분석에서는 심파이를 사용하여 미분, 적분을 한다. 손으로 계산하는 경우는 특별히 어떤 공식을 유도할 때만 한다.

#### `접선의 방정식`
- 단변수 함수 f(x)위의 한 점을 지나는 접선의 방정식
    - <img src="https://latex.codecogs.com/gif.latex?y-f%28a%29%3Df%5E%7B%5Cprime%7D%28a%29%28x-a%29"/>

