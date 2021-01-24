# 데이터분석과 함수

### 1. 함수

#### 데이터분석에서의 함수의 의미
- 입력데이터를 원하는 출력데이터로 만들어주는 **좋은 함수**를 찾는 것이 중요
    - `좋은 함수` : 성능이 떨어지는 **함수를 개선해서 좋은 함수로 만드는 과정**이 필요하다.
    - `미분, 적분` : 입력값이나 계수가 바뀌었을 때, 출력이 어떻게 변하는지를 알려주는 일종의 **신호**와 같다. 입출력 변화의 신호.
    - `범함수` : 함수 자체를 입력받아서 **함수의 점수**를 출력해준다.
    - 이외에도 미분과 적분, 부정적분, 정적분, 도함수, 행렬의 미분 등이 데이터분석에 활용된다.

#### 함수의 정의
- `함수 function` : 입력값을 출력값으로 바꾸어 출력하는 관계
    - 입력데이터를 받아서 출력데이터로 만들어 내보낸다.
    - `정의역 domain` : 입력값의 범위, 실수 전체인 경우가 대부분.
    - `공역 range` : 출력값의 범위
    - 1,2,3... → function → 2,4,6,...
    - **입력된 값에 대해서 항상 같은 출력값이 나와야 함수관계가 성립한다.**
- 함수관계는 실생활의 다양한 현상에 적용 된다. **서로 관계를 맺는 모든 숫자의 쌍은 함수이다.**
    - 엠프, 전자렌지, 선풍기, 보일러 등의 조절나사의 각도, 입력버튼의 누른 횟수 x 와 출력되는 결과물의 크기나 정도 y
    - 등산할 때 출발점에서 간 거리 x 와 그 위치의 고도 y
    - 자동차가 이동한 거리 x 와 연료 사용량 y 등.
- 정의역이 유한개의 원소로 이루어져 있으면 함수는 일종의 표 lookup table 이 된다.
    - 파이썬의 딕셔너리 {} 로 함수를 구현할 수 있다.

#### 변수
- `변수 variable` : 어떤 숫자를 대표하는 기호 sign. x, y, z 등으로 표기한다.
    - `입력변수 input variable` : 입력값을 대표하는 변수
    - `출력변수 ouput varialbe` : 출력값을 대표하는 변수
    - 변수와 수식을 사용하여 입출력값의 관계를 표현 할 수 있다.
- 함수의 표기 : f, g, h 등으로 표기한다.
    - x → 2x , y = 2x
    - f(x) = 2x
    - g(x) = 2x + 5
    - h(x) = 1/2x + 1

#### 연속과 불연속
- `불연속 discontinuous` : 함수의 값이 중간에 변하는 것.
- `연속 continuous` : 함수의 값이 변하지 않고 연속적인 것.
- `불연속 함수`
    - `부호함수 sign function` : 입력값이 양수이면 1, 0이면 0, 음수이면 -1 을 출력. x=0 에서 불연속인 함수.
        - <img src="https://latex.codecogs.com/gif.latex?sign%28x%29%20%3D%5Cbegin%7Bcases%7D%20-1%2C%20%26%20x%20%3C%200%20%5C%5C%200%2C%20%26%20x%20%3D%200%20%5C%5C%201%2C%20%26%20x%20%3E%200%20%5C%5C%20%5Cend%7Bcases%7D"/>
    - `단위계단 함수 heaviside step function` : 입력값이 0보다 크거나 같으면 1, 0보다 작으면 0 을 출력하는 함수.
        - <img src="https://latex.codecogs.com/gif.latex?H%28x%29%20%3D%20%5Cbegin%7Bcases%7D%201%2C%20%26%20x%20%5Cge%200%20%5C%5C%200%2C%20%26%20x%20%3C%200%20%5Cend%7Bcases%7D"/>
    - `지시함수 indicator function` : 미리 지정된 값이 입력되면 1, 지정된 값이 아니면 0을 출력하는 함수
        - <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%7BI%7D_i%28x%29%20%3D%20%5Cdelta_%7Bix%7D%20%3D%20%5Cbegin%7Bcases%7D%201%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3D%20i%20%5C%5C%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cneq%20i%20%5C%5C%20%5Cend%7Bcases%7D"/>
        - <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%7BI%7D%28x%3Di%29%20%3D%20%5Cbegin%7Bcases%7D%201%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3D%20i%20%5C%5C%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cneq%20i%20%5Cend%7Bcases%7D"/>
    - 지시함수는 데이터 중에서 특정한 데이터만 선택하여 그 갯수를 세는 데 사용되기도 한다.
        - <img src="https://latex.codecogs.com/gif.latex?N_0%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cmathbb%7BI%7D%28x_i%20%3D%200%29"/> (데이터의 값이 0인 것의 갯수)
        - <img src="https://latex.codecogs.com/gif.latex?N_1%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cmathbb%7BI%7D%28x_i%20%3D%201%29"/> (데이터의 값이 1인 것이 갯수)

#### 역함수
- `역함수 inverse function` : 어떤 함수의 입력,출력 관계와 **정반대의 입력,출력 관계를 갖는 함수.**
    - <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29%2C%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20x%20%3D%20f%5E%7B-1%7D%28y%29"/>
- 역함수 기호와 함수의 역수 기호는 유사하지만 혼동하며 안된다.
    - <img src="https://latex.codecogs.com/gif.latex?f%5E%7B-1%7D%28y%29%20%5Cneq%20f%28x%29%5E%7B-1%7D%20%3D%20%5Cfrac%20%7B1%7D%7Bf%28x%29%7D"/>
- `모든 함수가 항상 역함수를 갖는 것은 아니다.`
    - 서로 다른 입력값이 같은 출력값을 갖는 함수의 경우 역함수가 존재하지 않는다.
    - <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dx%5E2"/> 는 역함수가 존재하지 않는다. 그러나 x 의 범위를 양수로 제한하면 역함수가 존재한다. <img src="https://latex.codecogs.com/gif.latex?f%5E%7B-1%7D%28x%29%20%3D%20%5Csqrt%7Bx%7D"/>

### 2. 데이터분석에서 많이 사용되는 함수 10가지

#### `다항식 함수 polynomial function`
- 상수항 c0 와 거듭제곱 항의 선형조함으로 이루어진 함수
    - <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20c_0%20&plus;%20c_1x%20&plus;%20c_2x%5E2%20&plus;%20c_3x%5E3%20&plus;%20...%20&plus;%20c_nx%5En"/>

#### `최대함수와 최소함수`
- 두 개의 인수 중에서 큰 것 또는 작은 것을 출력해주는 함수
    - <img src="https://latex.codecogs.com/gif.latex?max%28x%2Cy%29%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%5Cgeq%20y%20%5C%5C%20y%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%3C%20y%20%5Cend%7Bcases%7D"/>
    - <img src="https://latex.codecogs.com/gif.latex?min%28x%2Cy%29%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%5Cleq%20y%20%5C%5C%20y%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%3E%20y%20%5Cend%7Bcases%7D"/>

#### `렐루 함수 ReLU rectified lenear unit`
- 최대함수에서 y 를 0 으로 고정시킨 함수. x가 양수이면 x, x가 음수이면 0 이 출력된다.
- 인공신경망에서 사용된다.
    - <img src="https://latex.codecogs.com/gif.latex?max%28x%2C0%29%3D%20%5Cbegin%7Bcases%7D%20x%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%5Cgeq%200%20%5C%5C%200%20%26%20%5Ctext%7Bif%7D%5C%3B%20x%20%3C%200%20%5Cend%7Bcases%7D"/>

#### `지수함수 exponential function`
- `밑 base 가 오일러 수 (약 2.718) 이고, 입력값을 거듭제곱으로 하는 함수`
- exponential : 기하급수적, 지수의
    - <img src="https://latex.codecogs.com/gif.latex?y%3De%5Ex"/>
    - <img src="https://latex.codecogs.com/gif.latex?y%3Dexp%28x%29%3Dexp%5C%3Bx"/>
- 넘파이에서 e 는 오일러수, exp() 명령은 지수함수를 계산해준다.
- `지수함수의 특징`
    - 오일러수는 양수이므로 거듭제곱한 수는 항상 양수이다.
    - x=0 일 때 1 이다.
    - x가 양의 무한대로 가면 y도 양의 무한대로 간다.
    - x가 음의 무한대로 가면 y는 0으로 다가간다.
    - x1 > x2 이면 exp x1 > exp x2 이다. (단조증가)
    - 두 지수함수의 곱은 입력값 합의 지수함수값과 같다.
        - <img src="https://latex.codecogs.com/gif.latex?f%28x_1%29%20%5Ccdot%20f%28x_2%29%20%3D%20e%5E%7Bx_1%7D%20%5Ccdot%20e%5E%7Bx_2%7D%20%3D%20e%5E%7Bx_1&plus;x_2%7D%20%3D%20f%28x_1&plus;x_2%29"/>

#### `로지스틱 함수 logistic function`
- 지수함수를 변형한 함수이다. 회귀분석이나 인공신경망 분야에서 자주 사용된다.
- 시그모이드 함수 sigmoid function 의 하나이다. 시그모이드 함수 중 대표적이므로 시그모이드 함수 = 로지스틱 함수로 쓰인다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Csigma%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;exp%28-x%29%7D"/>
- 로지스틱 함수의 특징
    - x가 양의 무한대로 갈 수록 y는 1에 가까워진다.
    - x가 음의 무한대로 갈 수록 y는 0에 가까워진다.
    - x가 0일때 y는 0.5이다.

#### `로그 함수 log function`
- 지수함수의 역함수이다.
- 지수함수의 출력이 특정한 값이 되게 하는 입력값을 찾는 것과 같다.
    - <img src="https://latex.codecogs.com/gif.latex?y%3De%5Ex%20%5C%3B%5C%3B%20a%3De%5Ex%20%5C%3B%5C%3B%20%5Clog%20a%20%3D%20x"/>
    - <img src="https://latex.codecogs.com/gif.latex?10%3De%5E%7B2.3025851%7D%20%5C%3B%5C%3B%20%5Clog%2010%20%3D%202.3025851"/>
    - <img src="https://latex.codecogs.com/gif.latex?y%3Dlogx"/>
- 지수함수의 특징
    - 입력값은 항상 양수여야 한다.
    - x > 1 → y > 0
    - x = 1 → y = 0
    - 0< x < 1 → y < 0
    - x1 > x2 → logx1 > logx2
- 넘파이 : np.log()

#### `로그 함수의 성질`
- `곱하기를 더하기로 변환한다.`
    - <img src="https://latex.codecogs.com/gif.latex?%5Clog%7B%28x_1%20%5Ccdot%20x_2%29%7D%20%3D%20%5Clog%7Bx_1%7D%20&plus;%20%5Clog%7Bx_2%7D"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Clog%7B%5Cleft%28%5Cprod_i%20x_i%20%5Cright%29%7D%20%3D%20%5Csum_i%20%28%5Clog%20x_i%29"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Clog%7Bx%5En%7D%3Dn%5Clog%7Bx%7D%24%2C%20%24%5Clog%7B%28x%20*%20x%20*%20x%29%7D%20%3D%20%5Clog%7Bx%7D%20&plus;%20%5Clog%7Bx%7D%20&plus;%20%5Clog%7Bx%7D%3D3%5Clog%7Bx%7D"/>
- `어떤 함수에 로그를 적용해도 함수의 최고점, 최저점의 위치는 변하지 않는다.`
    - 높낮이는 바뀌어도, 최고점, 최저점의 위치는 바뀌지 않는다.
    - 최적화 할 때 함수에 로그를 취해서 최적화를 하는 경우가 많다. <img src="https://latex.codecogs.com/gif.latex?%5Carg%5Cmax_x%20f%28x%29%3D%5Carg%5Cmax_x%20%5Clog%7Bf%28x%29%7D"/>
- `로그 함수는 0부터 1사이의 작은 값을 확대해서 보여준다.`
    - 입력값 0부터 1사이에 대하여 음의 무한대부터 0사이의 값으로 보여준다.
    - 따라서 확률값과 같이 0~1 사이의 **작은 값을 확대하여 비교를 잘 할 수 있게 해준다.**

#### `소프트 플러스 함수 softplus function`
- 지수함수와 로그함수를 결합하여 만든 함수.
- ReLU 함수와 유사하지만, x=0 에서 부드럽게 변하는 장점이 있다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Czeta%28x%29%20%3D%20%5Clog%7B%281%20&plus;%20%5Cexp%28x%29%29%7D"/>

#### `다변수 함수 multivariate function`
- 복수의 입력변수를 갖는 함수. **2차원 함수** 라고도 한다.
- 두 개의 독립변수 x, y 를 갖고, 출력변수 z 를 내보내는 함수.
    - <img src="https://latex.codecogs.com/gif.latex?z%3Df%28x%2Cy%29"/>
    - <img src="https://latex.codecogs.com/gif.latex?f%28x%2Cy%29%20%3D%202x%5E2%20&plus;%206xy%20&plus;%207y%5E2%20-%2026x%20-%2054y%20&plus;%20107"/>
- 다변수 함수의 대표적인 예로 평면상의 지형이 있다. 위도 x, 경도 y를 입력받아서 고도 z를 출력하는 함수이다.
- 서피스 플로 surface plot, 컨투어 플롯 contour plot 을 사용하여 나타낼 수 있다.

#### `분리가능 다변수 함수 separable multivariable function`
- 다변수 함수 중에는 **단변수의 곱**으로 나타낼 수 있는 함수들이 있다. 이러한 함수를 **분리가능 다변수 함수**라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?f%28x%2Cy%29%3Df_1%28x%29f_2%28y%29"/>
    - 확률론에서 중요하게 사용되는 함수이다.
- 2차원 함수는 지형도와 같으므로 x 또는 y 둘중 하나를 상수값으로 고정 시킬 수 있다. 이럴경우 움직일 수 있는 변수는 하나가 되어 1차원 단변수함수가 된다.
- **지형도의 단면의 모양과 같게된다.**
    - <img src="https://latex.codecogs.com/gif.latex?f%28x_0%2C%20y%29%3Df%28x_0%29f%28y%29%3Dk_0f%28y%29"/>
    - k0 는 고정된 값이므로 f(y) 만 조절한 모양이 된다.

#### `다변수 다출력 함수`
- **여러개의 변수를 받아서 여러개의 출력변수로 나타내주는 함수.** 벡터나 행렬로 출력할 수 있다.
- 소프트맥스 함수 softmax function : 다차원 벡터를 입력받아서 다차원 벡터를 출력해준다. 또한 **다변수 입력을 확률 처럼 보이게 출력해주는 특징**이 있어서 인공신경망의 마지막단에서 출력을 조건부로 변형하는데 사용된다. (벡터를 확률로 출력)
    - <img src="https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbegin%7Bbmatrix%7D%20y_1%20%5C%5C%20y_2%20%5C%5C%20y_3%20%5Cend%7Bbmatrix%7D%20%3DS%28x_1%2C%20x_2%2C%20x_3%29%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cexp%28w_1x_1%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cdfrac%7B%5Cexp%28w_2x_2%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cdfrac%7B%5Cexp%28w_3x_3%29%7D%7B%5Cexp%28w_1x_1%29%20&plus;%20%5Cexp%28w_2x_2%29%20&plus;%20%5Cexp%28w_3x_3%29%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>
- `출력벡터의 특징`
    - **모든 출력원소는 0과 1사이의 값을 갖는다.** : 지수함수값이 모두 양수이고, 분모가 분자보다 크기 때문이다.
    - **모든 출력원소의 합은 1이다.**
    - **입력원소의 크기 순서와 출력원소의 크기 순서가 같다.** : 단조증가 (x1 > x2 -> f(x1) > f(x2))

#### `함수의 이동`
- 그래프 상에서 함수를 이동시킬 수 있다.
- 단변수 함수의 이동
    - **x축 방향의 이동은 수식 적용시 부호 반대**
    - x축으로 +a 만큼 이동 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dx%5E2&plus;2x%20%5C%3B%5Crightarrow%5C%3B%20f%28x-a%29%3D%28x-a%29%5E2&plus;2%28x-a%29"/>
    - x축으로 -a 만큼 이동 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dx%5E2&plus;2x%20%5C%3B%5Crightarrow%5C%3B%20f%28x&plus;a%29%3D%28x&plus;a%29%5E2&plus;2%28x&plus;a%29"/>
    - **y축 방향의 이동은 수식 적용시 부호 같음**
    - y축으로 +b 만큼 이동 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5Crightarrow%5C%3B%20f%28x%29&plus;b"/>
    - y축으로 -b 만큼 이동 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5Crightarrow%5C%3B%20f%28x%29-b"/>
- 다변수 함수의 이동
    - **수식 적용시 부호가 반대**
    - x축으로 +a, y축으로 +b 이동 : <img src="https://latex.codecogs.com/gif.latex?f%28x%2Cy%29%20%5C%3B%5Crightarrow%5C%3B%20f%28x-a%2C%20y-b%29"/>

#### `함수의 스케일링`
- 단변수 함수의 스케일링
    - x축으로 a배 늘이기 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5Crightarrow%5C%3B%20f%28%5Cdfrac%7Bx%7D%7Ba%7D%29"/>
    - y축으로 b배 늘이기 : <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%5C%3B%5Crightarrow%5C%3B%20b%20%5Ccdot%20f%28x%29"/>

