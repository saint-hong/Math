# 적분, 행렬의 미분, 범함수

# 1. 적분

### 1) 부정적분
- `적분 integral` : 적분은 미분과 반대되는 개념이다. 적분에는 **부정적분 indefinite integral**, **적분 definite integral** 이 있다.
    - 부정적분 : 도함수의 원래 함수를 찾는 작업. 미분의 반대 과정.
    - 적분 : "정해져 있다.". 특정 구간에 해당하는 그래프의 면적을 구하는 작업.

#### `부정적분 indefinite integral`
- 부정적분은 **미분의 반대 anti-derivative** 개념이다. 도함수를 f(x)라고 할 때 도함수를 만들어 낸 원래 함수 F(x) 를 찾는 과정 또는 그 결과를 의미한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7BdF%28x%29%7D%7Bdx%7D%3Df%28x%29%5C%3B%5C%3B%5Cleftrightarrow%5C%3B%5C%3BF%28x%29%3D%5Cint%20f%28x%29dx%20&plus;%20C"/>
    - dx 는 x로 F를 x로 미분했다는 의미로 편미분에 대응하는 적분이라는 표기
    - C 는 상수항으로 미분을 하면 0이 되어 사라진 것을 의미한다. C에 대한 부정적분의 해는 무한개이다. 표기를 생략하기도 한다.

#### `편미분의 부정적분`
- 다변수 함수를 도함수라고 할 때 원래의 다변수 함수를 찾는 작업. 도함수의 결과가 다변수 함수라는 것은 원래의 함수 F도 다변수 함수이며 이것을 편미분한 결과라는 의미이다.
- 따라서 편미분 과정에서 변수 x 또는 y를 사용하기 때문에, 주어진 도함수 f를 적분할 때에도 x, y로 각각 적분해야 원래의 함수 F를 찾을 수 있다.
>- x로 부정적분 : <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20F_1%28x%2Cy%29%7D%7B%5Cpartial%20x%7D%3Df%28x%2Cy%29%5C%3B%5C%3B%5Cleftrightarrow%5C%3B%5C%3BF_1%28x%2Cy%29%3D%5Cint%20f%28x%2Cy%29dx%20&plus;%20C%28y%29"/>
>- y로 부정적분 : <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20F_2x%2Cy%29%7D%7B%5Cpartial%20y%7D%3Df%28x%2Cy%29%5C%3B%5C%3B%5Cleftrightarrow%5C%3B%5C%3BF_2%28x%2Cy%29%3D%5Cint%20f%28x%2Cy%29dy%20&plus;%20C%28x%29"/>
- C(x), C(y)는 상수항 또는 x, y의 함수일 수 있다. 원래의 함수 F를 한변수로 미분하면 다른변수는 상수항 처럼 취급되어 0이 되기 때문이다.

#### `다차도함수와 다중적분`
- 2차 도함수와 같이 어떤 함수를 미분을 여러번 하여 구한 다차도함수의 경우 원래의 함수를 구하기 위해선 다중적분 multiple integration 을 해야한다. 여러번 적분을 한다는 의미이다.
- 함수 f(x,y)가 F(x,y)를 x로 한번, y로 한번 편미분하여 나온 이차 도함수라고 할때, 원래의 함수 F를 구하기 위해서는 y로 적분을 하고 다시 x로 적분을 해야한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%5E2%20F_3%28x%2Cy%29%7D%7B%5Cpartial%20y%20%5Cpartial%20x%7D%3Df%28x%2Cy%29%5C%3B%5C%3B%5Cleftrightarrow%5C%3B%5C%3BF_3%28x%2Cy%29%3D%5Cint_%7Bx%7D%5Cint_%7By%7Df%28x%2Cy%29dydx"/>
>- 적분기호의 아래 변수명을 생략하여 쓰기도 한다 : <img src="https://latex.codecogs.com/gif.latex?%5Cint%5Cint%20f%28x%2Cy%29dydx"/>

#### `심파이를 이용한 부정적분`
- 단변수 함수 f를 적분 : sympy.integrate(f)
- 다변수 함수 f를 x로 적분 : sympy.integrate(f, x)
- 다변수 함수 f를 y로 적분 : sympy.integrate(f, y)

### 2) 정적분

#### `정적분 definite integral`
- 독립변수 x가 어떤 구간 [a,b]의 사이에 있을 때 **이 구간에서 f(x)의 값과 수평선 x축이 이루는 면적**을 구하는 행위 혹은 그 값을 의미한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7Ba%7D%5E%7Bb%7D%20f%28x%29dx"/>
- 정적분의 값은 숫자값이다.

#### `미적분학의 기본 정리 Fundamental Theorem Calculus`
- 함수 f의 특정 구간에서의 면적을 의미하는 정적분은 부정적분으로 구한 함수 F와 다음과 같은 관계가 성립하므로 미분과 관련이 있다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7Ba%7D%5E%7Bb%7D%20f%28x%29dx%20%3DF%28b%29-F%28a%29"/>
- 즉 함수 F로부터 미분을 하면 도함수 f를 구할 수 있고, 도함수 f를 부정적분하면 원래의 함수 F를 구할 수있다. 함수 F의 b와 a의 값의 차는 도함수 f의 a와 b 사이의 면적과 같다.

#### `심파이를 이용한 정적분`
- 함수 f를 부정적분하여 함수 F를 구하고, 이 F를 사용하여 정적분 값을 구하는 방법
- 함수 f의 a,b 구간의 면적을 실제로 잘개 쪼갠 후 근사값을 구하는 **수치적분 numerical integration** 방법

#### `다변수 정적분`
- 입력변수가 2개인 2차원 함수 f(x,y)의 정적분
- 2차원 함수 f(x,y)는 다변수 함수로 지형도와 같다. 위도 x, 경도 y 를 입력받아 고도 z 값을 출력하는 함수.
- 이러한 2차원 함수를 정적분하면 2차원 평면에서 사각형의 **부피**와 같다.
> <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7By%3Dc%7D%5E%7By%3Dd%7D%20%5Cint_%7Bx%3Da%7D%5E%7Bx%3Db%7D%20f%28x%2Cy%29dxdy"/>

#### `수치이중적분`
- 단변수 함수의 정적분에서 수치적분 처럼 다변수함수의 정적분도 수치이중적분이 가능하다.
- 사이파이의 integrate 서브패키지의 dblquad() 명령을 사용한다.
    - sp.integrate.dblquad(func, a, b, gfun, hfun)
    - a, b : x의 범위, **x의 하한 lower bound, x의 상한 upper bound**
    - gfun, hfun : y의 범위. 함수 형태로 입력받는다. (lambda x : a, lambda x : b)

#### `다차원 함수의 단일정적분`
- 다차원 함수이지만 변수를 하나만 선택하여 정적분을 할 수도 있다. 이때에는 하나의 변수만 진짜로 보고 다른 변수는 상수항 처럼 취급하여 계산한다.
> <img src="https://latex.codecogs.com/gif.latex?%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%2Cy%29dx"/>
- y가 변수가 아니라는 것을 강조한 표기 : <img src="https://latex.codecogs.com/gif.latex?f%28x%3By%29%20%3D%204x%5E2%20&plus;%20%284y%29x%20&plus;%20%28y%5E2%29"/>

# 2. 행렬의 미분

### 1) 행렬미분

#### `행렬미분 matrix differentiation`
- 벡터나 행렬을 입력 받아서 벡터나 행렬로 출력해주는 함수의 미분
- 여러가지 입력변수와 출력변수의 종류
    - 다변수 함수는 함수의 독립변수가 벡터인 경우이다.
    - 벡터 x -> 스칼라 f
    > <img src="https://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7Dx_1%5C%5Cx_2%5Cend%7Bbmatrix%7D%5Cright%29%20%3D%20f%28x%29%20%3D%20f%28x_1%2C%20x_2%29"/>

    - 행렬 x -> 스칼라 f
    > <img src="https://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7Dx_%7B11%7D%20%26%20x_%7B12%7D%5C%5Cx_%7B21%7D%20%26%20x_%7B22%7D%5Cend%7Bbmatrix%7D%5Cright%20%29%20%3D%20f%28x%29%20%3D%20f%28x_%7B11%7D%2C%20x_%7B12%7D%2C%20x_%7B21%7D%2C%20x_%7B22%7D%29"/>

    - 벡터나 행렬을 출력하는 함수는 여러 함수를 합쳐놓은 것으로 볼 수 있다.
    - 스칼라 x -> 벡터 f
    > <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cbegin%7Bbmatrix%7Df_1%28x%29%5C%5Cf_2%28x%29%5Cend%7Bbmatrix%7D"/>

    - 스칼라 x -> 행렬 f
    > <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cbegin%7Bbmatrix%7Df_%7B11%7D%28x%29%26f_%7B12%7D%28x%29%5C%5Cf_%7B21%7D%28x%29%26f_%7B22%7D%28x%29%5Cend%7Bbmatrix%7D"/>

    - 벡터나 행렬을 입력받아 벡터나 행렬을 출력할 수 있다.
    - 벡터 x -> 벡터 f
    > <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cbegin%7Bbmatrix%7Df_%7B1%7D%28x_1%2C%20x_2%29%5C%5Cf_%7B2%7D%28x_1%2C%20x_2%29%5Cend%7Bbmatrix%7D"/>

    - 벡터 x -> 행렬 f
    > <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cbegin%7Bbmatrix%7Df_%7B11%7D%28x_1%2C%20x_2%29%26f_%7B12%7D%28x_1%2C%20x_2%29%5C%5Cf_%7B21%7D%28x_1%2C%20x_2%29%26f_%7B22%7D%28x_1%2C%20x_2%29%5Cend%7Bbmatrix%7D"/>
- 행렬미분은 편미분의 일종이다.
- 행렬미분의 종류
    - 분자중심 표현법 Numerator-layout notation
    - 분모중심 표현법 Denominator-layout notation

#### `그레디언트  벡터 gadient vector`
- 데이터분석에서는 출력변수가 스칼라이고 입력변수가 x 벡터인 다변수함수인 경우가 많다. 즉 x의 요소들로 편미분을 해줘야 하므로 여러개의 편미분이 존재한다.
- **스칼라를 벡터로 미분하는 경우**에는 열벡터로 표기한다. 이렇게 만들어진 벡터를 **그레디언트 벡터** 라고 한다.
> 그레디언트 벡터 : <img src="https://latex.codecogs.com/gif.latex?%5Cnabla%20f%20%3D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%5C%5C%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%5C%5C%20%5Ccdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_N%7D%20%5Cend%7Bbmatrix%7D"/>
- 그레디언트 벡터는 기울기로 이루어진 열벡터 이다.
- 2차원 함수, 다변수 함수 f(x,y)는 경도 x, 위도 y에 따라 출력되는 고도 z의 지형도와 같다. f(x,y)의 그레디언트 벡터.
> <img src="https://latex.codecogs.com/gif.latex?%5Cnabla%20f%20%3D%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D%5Cend%7Bbmatrix%7D"/>
- 즉 다변수 함수를 행렬미분한 값인 그레디언트 벡터는 x로 미분한 도함수와 y료 미분한 도함수로 이루어져 있다.
- 그레디언트 벡터의 특징
    - **그레디언트 벡터의 크기는 함수 곡면의 기울기의 정도를 나타낸다.** 벡터의 크기가 클 수록 경사가 심하다는 것을 의미한다.
    - **그레디언트 벡터의 방향은 가장 경사가 심한 곳을 가리킨다.** 즉 단위 길이당 함숫값이 가장 크게 증가하는 방향을 가리킨다.
    - **그레디언트 벡터의 방향은 등고선의 방향과 직교한다.**
- **테일러전개식**을 사용하여 그레디언트 벡터의 방향과 등고선의 방향이 직교함을 증명할 수 있다.
>- 어떤 점 x0 에서 x로 이동하면서 함숫값이 얼마나 변하는지 테일러 전개를 사용하여 근사하면 다음과 같다.
>>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20-%20f%28x_0%29%20%3D%20%5CDelta%20f%20%5Capprox%20%5Cnabla%20f%28x_0%29%5ET%28x-x0%29"/>
>- 이 식에서 <img src="https://latex.codecogs.com/gif.latex?%5CDelta%20f"/> 가 가장 크려면, 변화의 방향인 (x-x0)와 그레디언트 벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cnabla%20f"/>의 방향이 같아야 한다. 곱해서 양수.
>- 또한 등고선에서는 x의 위치에 상관없이 같은 높이이므로 함수값이 같다.
>>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Df%28x_0%29%24%2C%20%24f%28x%29-f%28x_0%29%3D0"/>
>- 등고선의 방향과 그레디언트 벡터의 방향을 테일러전개식에서 정리하면,
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x_0%29%5ET%28x-x0%29%3Df%28x_1%29-f%28x_0%29%3D0"/>
>>- <img src="https://latex.codecogs.com/gif.latex?a%5ET%20%5Ccdot%20b%3D0%5C%3B%5C%3B%5Cleftrightarrow%5C%3B%5C%3Ba%20%5Cperp%20b"/>

### 2) 행렬미분법칙

#### `행렬미분법칙 1 : 선형모형의 미분`
- 선형조합으로 이루어진 함수를 미분하면 그레디언트 벡터는 **가중치 벡터 w** 이다.
>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20w%5ETx%24%2C%20%24%5Cnabla%20f%20%3D%20%5Cdfrac%7B%5Cpartial%20%28w%5ETx%29%7D%7B%5Cpartial%20x%7D%3D%5Cdfrac%7B%5Cpartial%20%28x%5ETw%29%7D%7B%5Cpartial%20x%7D%3Dw"/>
- x는 벡터이므로 x의 각 요소들 x1,x2,x3,..,xn 로 w^Tx 를 각각 미분한다. 선형조합은 w1x1+w2x2+...+wnxn 이므로 x의 요소에 **해당하지 않는 항은 미분하여 0이되어 없어지고, x의 요소에 해당하는 값들만 미분이되어 w1, w2, w3... 만 남게 된다.** 따라서 가중치 벡터가 된다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20%28%7Bw%7D%5ET%20%7Bx%7D%29%7D%7B%5Cpartial%20%7Bx%7D%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28%7Bw%7D%5ET%20%7Bx%7D%29%7D%7B%5Cpartial%20x_1%7D%7D%20%5C%5C%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28%7Bw%7D%5ET%20%7Bx%7D%29%7D%7B%5Cpartial%20x_2%7D%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28%7Bw%7D%5ET%20%7Bx%7D%29%7D%7B%5Cpartial%20x_N%7D%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28w_1%20x_1%20&plus;%20%5Ccancel%7Bw_2%20x_2%7D%20&plus;%20%5Ccdots%20&plus;%20%5Ccancel%7Bw_N%20x_N%7D%29%7D%7B%5Cpartial%20x_1%7D%20%5Cmathstrut%7D%20%5C%5C%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28%5Ccancel%7Bw_1%20x_1%7D%20&plus;%20w_2%20x_2%20&plus;%20%5Ccdots%20&plus;%20%5Ccancel%7Bw_N%20x_N%7D%29%7D%7B%5Cpartial%20x_2%7D%20%5Cmathstrut%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20%5Cboxed%7B%5Cdfrac%7B%5Cpartial%20%28%5Ccancel%7Bw_1%20x_1%7D%20&plus;%20%5Ccancel%7Bw_2%20x_2%7D%20&plus;%20%5Ccdots%20&plus;%20w_N%20x_N%29%7D%7B%5Cpartial%20x_N%7D%20%5Cmathstrut%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20w_1%20%5C%5C%20w_2%20%5C%5C%20%5Cvdots%20%5C%5C%20w_N%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%7Bw%7D"/>

#### `행렬미분법칙 2 : 이차형식의 미분`
- 이차형식 Quadratic : 벡터의 이차형식이란 어떤 벡터와 정방행렬이 **행벡터X정방행렬X열벡터** 형식으로 되어 있는 것을 의미한다.
>- 여러개의 벡터에 대하여 가중합을 동시에 계산
>> <img src="https://latex.codecogs.com/gif.latex?%5Chat%20y%3DXw"/>
>- 잔차 = 목표값 - 예측값
>> <img src="https://latex.codecogs.com/gif.latex?e%3Dy-%5Chat%20y%3Dy-Xw"/>
>- 잔차 제곱합 RSS(Residual Sum of Squares) : 선형회귀모델의 성능을 평가할 수 있는 잔차의 크기는 잔차 제곱합으로 구할 수 있다. 잔차 제곱합은 잔차의 내적으로 간단하게 표현할 수 있다.
>> <img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5E%7BN%7De_i%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i-w%5ETx_i%29%5E2%3De%5ETe%3D%28y-Xw%29%5ET%28y-Xw%29%3Dy%5Ey-w%5ETX%5ETy-y%5EXw&plus;w%5ETX%5ETXw"/>
>- 잔차제곱합의 식에서 마지막항을 벡터의 이차형식이라고 한다. 이차형식은 i, j 의 모든 쌍의 조합에 가중치를 곱한 값들의 총합이 된다.(스칼라)
>> <img src="https://latex.codecogs.com/gif.latex?w%5ETX%5ETXw%3Dx%5ETAx%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Csum_%7Bj%3D1%7D%5E%7BN%7Da_%7Bi%2Cj%7Dx_ix_j%3Dscala"/>
- 이차형식을 미분하면 행렬과 벡터의 곱이 된다
> <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dx%5ETAx%2C%5C%3B%5Cnabla%20f%28x%29%3D%5Cdfrac%7B%5Cpartial%20x%5ETAx%7D%7B%5Cpartial%20x%7D%3DAx&plus;A%5ETx%3D%28A&plus;A%5ET%29x"/>
- 선형조합을 벡터 x의 각 요소들로 미분을 하는 과정에서 선형조합을 aijxixj의 이중합으로 변환하면 현재 미분하려는 x의 요소에 대응하는 항만 남고 해당하지 않는 항은 0이되어 사라진다. 이 과정은 마치 행렬식의 여인수 전개에서 코펙터를 구하기 위해 i,j 번째를 제외하고 나머지 요소들을 제거하는 방식과 반대의 방식과 같다. (행렬식은 역행렬을 구할 때 사용된다. 역행렬은 행렬식이 0이 아닌 경우에만 존재한다)

#### `벡터를 스칼라로 미분`
- 함수 f1, f2,...,fn 을 벡터로 갖는 f(x)를 스칼라 x로 미분하는 경우에는 행벡터로 결과를 나타낼 수 있다.
>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cbegin%7Bbmatrix%7D%20f_1%5C%5Cf_2%5C%5C%20%5Cvdots%20%5C%5C%20f_M%20%5Cend%7Bbmatrix%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20%7Bx%7D%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20%7Bx%7D%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20%7Bx%7D%7D%20%5Cend%7Bbmatrix%7D"/> <img src="https://latex.codecogs.com/gif.latex?%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_M%7D%20%5Cend%7Bbmatrix%7D%20%3D"/> <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_M%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_M%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_M%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>

#### `벡터를 벡터로 미분`
- 벡터를 입력으로 받아 벡터를 출력하는 함수 f(x)의 미분은 **도함수의 2차원 행렬**이 된다.
- 벡터를 벡터로 미분하면, 미분을 당하는 원소 N개와 미분을 하는 벡터의 원소 M개 이다.
> <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20%7Bx%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20%7Bx%7D%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20%7Bx%7D%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20%7Bx%7D%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20%7Bf%7D%7D%7B%5Cpartial%20x_M%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_M%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_M%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_N%7D%7B%5Cpartial%20x_M%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>

#### `행렬미분법칙 3 : 행렬과 벡터의 곱의 미분`
- 행렬 A와 벡터 x의 곱 Ax를 벡터 x로 미분하면 행렬 <img src="https://latex.codecogs.com/gif.latex?A%5ET"/> 가 된다.
>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3DAx%2C%5C%3B%5Cnabla%20f%28x%29%3D%5Cdfrac%7B%5Cpartial%20Ax%7D%7B%5Cpartial%20x%7D%3DA%5ET"/>
>- 행렬과 벡터의 곱 정리에서 A를 열벡터들 c1, c2,...,cM 의 조합이라고 보면 Ax는 선형조합의 형태가 된다.
>>- <img src="https://latex.codecogs.com/gif.latex?Ax%20%3D%20c_1x_1&plus;c_2x_2&plus;c_3x_3&plus;%5Ccdots&plus;c_Mx_M"/>
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%28Ax%29%7D%7B%5Cpartial%20x%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%28Ax%29%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%28Ax%29%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Ccdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%28Ax%29%7D%7B%5Cpartial%20x_M%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%28c_1x_1&plus;c_2x_2&plus;c_3x_3&plus;%5Ccdots&plus;c_Mx_M%29%5ET%7D%7B%5Cpartial%20x_1%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%28c_1x_1&plus;c_2x_2&plus;c_3x_3&plus;%5Ccdots&plus;c_Mx_M%29%5ET%7D%7B%5Cpartial%20x_2%7D%20%5C%5C%20%5Ccdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%28c_1x_1&plus;c_2x_2&plus;c_3x_3&plus;%5Ccdots&plus;c_Mx_M%29%5ET%7D%7B%5Cpartial%20x_M%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20c_1%5ET%20%5C%5C%20c_2%5ET%20%5C%5C%20%5Ccdots%20%5C%5C%20c_M%5ET%20%5Cend%7Bbmatrix%7D%3D%20A%5ET"/>
>>- 미분하는 변수인 x1, x2,...,xm에 대응하는 선형조합의 항만 남고 나머지 항은 상수항 처리되어 0으로 없어진다.

#### `자코비안 행렬 Jacobian matrix`
- 벡터를 벡터로 미분하는 경우와 같이 벡터를 입력받아 벡터를 출력하는 함수의 경우, 입력변수 요소 각각과 출력변수 요소 각각의 조합에 대해 모두 미분이 존재한다. 따라서 이러한 경우는 도함수가 행렬 형태가 된다.
    - 다차원 데이터의 미분, 벡터를 입력 받아서 벡터를 출력하는 함수를 미분하면 도함수의 행렬 형태가 된다.
- 이러한 **도함수의 행렬 형태를 자코비안 행렬 Jacobian matrix** 이라고 한다.
- 또한 벡터함수를 벡터 변수로 미분했을 때 생기는 행렬의 **전치행렬**이다. NXM -> MXN 의 형태가 된다..
> <img src="https://latex.codecogs.com/gif.latex?Jf%28x%29%3DJ%3D%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%5Cright%29%5ET%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x%7D%5Cright%29%5ET%20%5C%5C%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x%7D%5Cright%29%5ET%20%5C%5C%20%5Ccdots%20%5C%5C%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20f_M%7D%7B%5Cpartial%20x%7D%5Cright%29%5ET%20%5C%5C%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cnabla%20%7Bf_1%7D%5ET%20%5C%5C%20%5Ccdots%20%5C%5C%20%5Cnabla%20%7Bf_M%7D%5ET%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_N%7D%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_M%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f_M%7D%7B%5Cpartial%20x_N%7D%20%5Cend%7Bbmatrix%7D"/>
- 다음 함수의 자코비안 행렬을 구하시오.
>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3D%20%5Cbegin%7Bbmatrix%7D%20%5Csum_%7Bi%3D1%7D%5E%7B3%7Dx_i%20%5C%5C%20%5Cprod_%7Bi%3D1%7D%5E%7B3%7Dx_i%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?Jf%28x%29%3DJ%3D%5Cleft%28%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%5Cright%29%5ET%3D%20%5Cbegin%7Bbmatrix%7D%20%28%5Cnabla%20f_1%29%5ET%20%5C%5C%20%28%5Cnabla%20f_2%29%5ET%20%5C%5C%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_3%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_3%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?%3D%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%28x_1&plus;x_2&plus;x_3%29%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%28x_1&plus;x_2&plus;x_3%29%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%28x_1&plus;x_2&plus;x_3%29%7D%7B%5Cpartial%20x_3%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%28x_1x_2x_3%29%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%28x_1x_2x_3%29%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cdfrac%7B%5Cpartial%28x_1x_2x_3%29%7D%7B%5Cpartial%20x_3%7D%20%5C%5C%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%201%20%26%201%20%5C%5C%20x_2x_3%20%26%20x_1x_3%20%26%20x_1x_2%20%5C%5C%20%5Cend%7Bbmatrix%7D"/>

#### `헤시안 행렬 Hessian matrix`
- 다변수함수의 2차 도함수는 그레디언트 벡터를 입력변수 벡터로 미분한 것으로 **헤시안 행렬** 이라고 한다.
- 헤시안 행렬은 그레디언트 벡터의 자코비안 행렬의 전치행렬로 정의한다.
>- <img src="https://latex.codecogs.com/gif.latex?Hf%28x%29%3DH%3DJ%28%5Cnabla%20f%28x%29%29%5ET%3DH_%7Bi%2Cj%7D%3D%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_i%20%5Cpartial%20x_j%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?H%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_1%5E2%7D%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_1%5C%2C%5Cpartial%20x_2%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_1%5C%2C%5Cpartial%20x_N%7D%20%5C%5C%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_2%5C%2C%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_2%5E2%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_2%5C%2C%5Cpartial%20x_N%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_N%5C%2C%5Cpartial%20x_1%7D%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_N%5C%2C%5Cpartial%20x_2%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%5E2%20f%7D%7B%5Cpartial%20x_N%5E2%7D%20%5Cend%7Bbmatrix%7D"/>
- 함수가 연속이고 미분가능하면 헤시안 행렬은 **대칭행렬**이 된다.
    - 대칭행렬 : <img src="https://latex.codecogs.com/gif.latex?A%3DA%5ET"/>
- 어떤 함수의 헤시안 행렬을 구하려면 그레디언트 벡터를 구하고, 자코비안 행렬을 구한 뒤, 전치연산한 행렬을 찾는다.
    - 그레디언트 벡터는 미분의 대상인 함수는 동일하고 미분할 변수가 여러개로, 함수안에 해당하는 변수가 있는 경우만 남게되어 간단하게 형태로 정리된다.
    - 자코비안 행렬은 미분의 대상인 함수가 여러개이고 미분할 변수도 여러개이므로, 함수 1 과 미분 변수 여러개, 함수 2와 미분변수 여러개 의 형태이다. 즉 모든 입출력변수의 조합에 대한 도함수가 행렬의 형태로 만들어 진다. 마찬가지로 각 함수에 해당하는 변수가 있는 경우만 남게 되고 없는 경우는 상수항 처리되어 없어진다. 자코비안 행렬에서 상수항만 남게 된다.
    - 자코비안 행렬을 구한 뒤 전치연산으로 형태를 바꿔주면 헤시안 행렬이 된다.

#### `스칼라를 행렬로 미분`
- 출력변수 f가 스칼라이값이고 입력변수 X가 행렬인 경우, 행렬로 스칼라를 미분하면 도함수의 행렬의 모양이 입력변수 X의 모양과 같다.
> <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20X%7D%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B1%2C1%7D%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B1%2C2%7D%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B1%2CN%7D%7D%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B2%2C1%7D%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B2%2C2%7D%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7B2%2CN%7D%7D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%5C%5C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7BM%2C1%7D%7D%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7BM%2C2%7D%7D%20%26%20%5Ccdots%20%26%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_%7BM%2CN%7D%7D%5C%5C%20%5Cend%7Bbmatrix%7D"/>
- 행렬 X의 모든 원소들로 F를 미분하는 방식으로 X의 행렬 구조와 모양이 같다.

#### `행렬미분법칙 4 : 행렬 곱의 대각성분의 미분`
- 두 정방행렬의 곱으로 만들어진 행렬의 대각성분의 합을 곱한 순서상 뒤의 행렬로 미분하면, 앞의 행렬의 전치행렬이 된다.
- 대각합 trace : 행렬의 대각성분의 합. 행렬식, 놈과 함께 행렬의 크기를 계산하는 방법 중 하나.
>- <img src="https://latex.codecogs.com/gif.latex?f%28x%29%3Dtr%28WX%29%2C%5C%3B%5C%3B%20W%20%5Cin%20R%5E%7BN%20%5Ctimes%20N%7D%2C%5C%3B%5C%3B%20X%20%5Cin%20R%5E%7BN%20%5Ctimes%20N%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20X%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20tr%28WX%29%7D%7B%5Cpartial%20X%7D%3DW%5ET"/>
>>- <img src="https://latex.codecogs.com/gif.latex?tr%28WX%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Csum_%7Bj%3D1%7D%5E%7BN%7Dw_%7Bj%2Ci%7Dx_%7Bi%2Cj%7D"/>
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20tr%28WX%29%7D%7B%5Cpartial%20x_%7Bi%2Cj%7D%7D%3Dw_%7Bj%2Ci%7D"/>
- 행렬의 대각합의 요소들을 행렬 X의 요소들로 미분하면, X의 요소에 해당하는 대각합의 요소들은 미분이되어 w_i,j 만 남고, 해당하지 않는 요소들은 상수항 처리되어 0이 된다.
- 그러므로 남는 값은 w_i,j 즉 처음 행렬의 전치행렬이 된다.

#### `행렬미분법칙 5 : 행렬식의 로그의 미분`
- 행렬식 determinant 은 행렬의 크기에 해당한다. det(A), |A| 등으로 표기하며, 재귀적인 방식으로 계산된다. 즉 행렬을 정의하기 위해선 행렬 자신이 필요하다는 의미이다.
- 행렬식의 공식은 여인수 전개 cofactor expansion 라는 식을 말한다. i_0 또는 j_0 중 하나를 정한 후, 부호결정과 마이너(처음 행렬 소거후 남은 요소들로 만든 행렬의 행렬식)의 곱으로 이루어진 코펙터(=여인수)와 a_i0,j 번째 요소의 곱이다.
>- <img src="det(A) = |A| = \sum_{i=1}^{N} \left\{ (-1)^{i+j_0}M_{i,j_0} \right\} a_{i,j_0}">
>- <img src="https://latex.codecogs.com/gif.latex?C_%7Bi%2Cj%7D%20%3D%20%28-1%29%5E%7Bi&plus;j_0%7DM_%7Bi%2Cj_0%7D"/>
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdet%28A%29%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20C_%7Bi%2Cj_0%7D%20a_%7Bi%2Cj_0%7D%20%3D%20%5Csum_%7Bj%3D1%7D%5EN%20C_%7Bi_0%2Cj%7D%20a_%7Bi_0%2Cj%7D"/>
- 행렬식의 계산 결과는 스칼라이고, 이 값의 로그값도 스칼라이다. 이 값을 원래의 행렬로 미분하면 원래 행렬의 역행렬의 전치행렬이 된다.
>- <img src="https://latex.codecogs.com/gif.latex?f%28X%29%20%3D%20%5Clog%20%7C%20%7BX%7D%20%7C%2C%5C%3B%5C%3B%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20X%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20%5Clog%20%7C%20%7BX%7D%20%7C%20%7D%7B%5Cpartial%20%7BX%7D%7D%20%3D%20%28%7BX%7D%5E%7B-1%7D%29%5ET"/>
>- 행렬식을 X 행렬의 요소 x_i,j로 미분하면, 행렬식의 정의에서 a_i,j 가 지워지고 C_i,j만 남는다.
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20x_%7Bi%2Cj%7D%7D%20%5Cvert%20X%20%5Cvert%20%3D%20C_%7Bi%2Cj%7D"/>
>- 따라서 행렬 X로 행렬식을 미분하면 C 가 된다. 역행렬의 정의에서 C는 다음과 같이 도출된다.
>>- <img src="https://latex.codecogs.com/gif.latex?A%5E%7B-1%7D%3D%5Cdfrac%7B1%7D%7Bdet%28A%29%7DC%5ET"/>
>>- <img src="https://latex.codecogs.com/gif.latex?C%3D%7CA%7C%28A%5E%7B-1%7D%29%5ET%2C%5C%3B%5C%3B%28%7CA%7C%5ET%3Dscala%29"/>
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20X%7D%20%5Cvert%20X%20%5Cvert%20%3D%20C%20%3D%20%7C%20X%20%7C%20%28X%5E%7B-1%7D%29%5ET"/>
>- 이것을 로그 함수의 미분 공식에 대입하면 원래행렬 X의 역행렬의 전치행렬이 된다.
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%20%5Clog%20f%28x%29%20%3D%20%5Cdfrac%7Bf%27%28x%29%7D%7Bf%28x%29%7D%20%3D%20%5Cdfrac%7B%5Cvert%20X%20%5Cvert%20%28X%5E%7B-1%7D%29%5ET%7D%7B%5Cvert%20X%20%5Cvert%7D%20%3D%20%28X%5E%7B-1%7D%29%5ET"/>
>>- 로그 함수의 미분은 연쇄법칙을 사용하여 풀이된다.

# 3. 범함수

### 1) 범함수 functional
- 데이터분석에서는 함수와 더불어 다양한 범함수를 사용한다.
- 범함수 functional 는 **함수를 입력받아서 실수를 출력해준다.** (함수 function 은 실수를 입력받아서 실수를 출력한다.)
    - 기댓값, 엔트로피 등을 계산할 때 사용된다.

#### ```범함수의 표기```
- 보통 알파벳 대문자로 표기하고, 입력변수인 함수를 대괄호로 감싼다.
>- <img src="https://latex.codecogs.com/gif.latex?F%5Cleft%5By%28x%29%5Cright%5D"/> : 함수 y(x) 를 입력받는 다는 의미

#### ```범함수의 계산```
- 일반적으로 범함수는 **정적분 definite integral 로 계산한다.**
- 확률밀도함수 p(x) 를 정적분한 값으로 갖는 기댓값과 엔트로피도 범함수에 속한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20E%5Bp%28x%29%5D%20%3D%20%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7D%20xp%28x%29dx"/>
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20H%5Bp%28x%29%5D%20%3D%20-%20%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7D%20p%28x%29%20%5Clog%20p%28x%29dx"/>

#### ```변분법 functional calculus```
- 입력변수인 함수가 변하는 것에 따라 범함수의 출력이 어떻게 달라지는를 계산하는 학문을 의미한다.
- 미분과 같은 기능이다.

### 2) 범함수의 테일러 전개

#### ```테일러 전개```
- 테일러 전개 : 함수 f(x) 의 도함수를 사용하여 함수 f(x) 의 근삿값을 구하는 방법
- 함수 -> 단변수 함수 -> 다변수 함수 -> 변수를 수열로 변환 -> 수열을 함수로 변환 -> 범함수 의 순서대로 테일러 전개식을 확장 해본다.
- **일반 함수** f(x) 에 대한 테일러 전개식
>- <img src="https://latex.codecogs.com/gif.latex?f%28x&plus;%5Cepsilon%29%20%5Capprox%20f%28x%29%20&plus;%20%5Cdfrac%7Bdf%7D%7Bdx%7D%5Cepsilon"/>  (<img src="https://latex.codecogs.com/gif.latex?%5Cepsilon"/> 은 아주 작은 실수)
- **단변수 함수** F(y)의 테일러 전개식, 일반함수의 테일러 전개식에서 표기문자만 바꿈. 입력변수는 y 라는 실수값.
>- <img src="https://latex.codecogs.com/gif.latex?F%28y&plus;%5Cepsilon%29%20%5Capprox%20F%28y%29%20&plus;%20%5Cdfrac%7BdF%7D%7Bdy%7D%5Cepsilon"/>
- **다변수 함수** F(y1,y2,...,yN)일 경우의 테일러 전개식. 입력변수는 y1,y2,...,yN N개의 실수
>- <img src="https://latex.codecogs.com/gif.latex?F%28y_1&plus;%5Cepsilon_1%2C%20y_2&plus;%5Cepsilon_2%2C%20%5Ccdots%2C%20y_N&plus;%5Cepsilon_N%29%20%5Capprox%20F%28y_1%2C%20y_2%2C%20%5Ccdots%2C%20y_N%29%20&plus;%20%5Cdfrac%7B%5Cpartial%20F%7D%7B%5Cpartial%20y_1%7D%5Cepsilon_1%20&plus;%20%5Cdfrac%7B%5Cpartial%20F%7D%7B%5Cpartial%20y_2%7D%5Cepsilon_2&plus;%20%5Ccdots%5C%20&plus;%5Cdfrac%7B%5Cpartial%20F%7D%7B%5Cpartial%20y_N%7D%5Cepsilon_N"/>
>- <img src="https://latex.codecogs.com/gif.latex?%3DF%28y_1%2C%20y_2%2C%20%5Ccdots%2C%20y_N%29%20&plus;%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cdfrac%7B%5Cpartial%20F%7D%7B%5Cpartial%20y_i%7D%5Cepsilon_i"/>
- **조건 추가**
    - yi 는 xi 를 입력받아 출력된 어떤 값 : <img src="https://latex.codecogs.com/gif.latex?y_i%3Dy%28x_i%29"/>
    - 엡실론i 는 xi 를 입력받는 임의의 함수에 아주 작은 공통상수를 곱한 값 : <img src="https://latex.codecogs.com/gif.latex?%5Cepsilon_i%3D%5Cepsilon%20%5Ceta%20%28x_i%29"/>
>- <img src="https://latex.codecogs.com/gif.latex?F%28y%28x_1%29&plus;%5Cepsilon%20%5Ceta%20%28x_1%29%2Cy%28x_1%29&plus;%5Cepsilon%20%5Ceta%20%28x_1%29%2C%5Ccdots%2Cy%28x_N%29&plus;%5Cepsilon%20%5Ceta%20%28x_N%29%29%20%5Capprox%20F%28y%28x_1%29%2C%20y%28x_2%29%2C%20%5Ccdots%2C%20y%28x_N%29%29%20&plus;%20%5Cepsilon%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cdfrac%7B%5Cpartial%20F%7D%7B%5Cpartial%20y_i%7D%20%5Ceta%20%28x_i%29"/>
- 위의 식은 벡터나 수열을 입력받아 실수를 출력하는 함수의 테일러 전개와 같다고 볼 수 있다.
    - 수열 {y(x1), y(x2), ..., y(xN)} -> 함수 F({y(x1), y(x2), ..., y(xN)})
- 수열의 크기 N 을 무한대로 확장한다면 수열은 함수 f(x) 를 의미하게 된다. 따라서 위의 다변수 함수의 테일러 전개식에서 조건 2가지를 추가하여 정리한 식은 이제 **범함수 F[y(x)] 에대한 테일러 전개식이 된다.**
    - 함수 y(x) -> 함수 F[y(x)]
>- <img src="https://latex.codecogs.com/gif.latex?F%5By%28x%29&plus;%5Cepsilon%20%5Ceta%20%28x%29%5D%20%5Capprox%20F%5By%28x%29%5D%20&plus;%20%5Cepsilon%20%5Cint%20%5Cdfrac%7B%5Cdelta%20F%7D%7B%5Cdelta%20y%28x%29%7D%20%5Ceta%28x%29dx"/>

### 3) 범함수의 도함수

#### ```범함수의 도함수```
- 범함수의 테일러 전개식에서 상수의 변화에 따른 범함수 값의 변화. 위의 식을 정리한 것.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7BF%5By%28x%29&plus;%5Cepsilon%20%5Ceta%28x%29%5D-F%5By%28x%29%5D%7D%7B%5Cepsilon%7D%20%3D%20%5Cint%20%5Cdfrac%7B%5Cdelta%20F%7D%7B%5Cdelta%20y%28x%29%7D%20%5Ceta%28x%29dx"/>
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20F%7D%7B%5Cdelta%20y%28x%29%7D%3D0"/> (어떤 함수 에타에 대해서도 이 값이 0 이 되려면)
- 따라서 이것을 **범함수의 도함수 functional derivative** 라고 한다. 일반함수의 도함수와 같은 역할을 한다. 범함수 F 를 미분했다는 의미로 델다 표기를 사용함.

#### ```적분형 범함수의 도함수```
- 대부분의 범함수는 x 에 대한 적분으로 정의된다. G 는 함수 y(x), 실수 x 를 입력변수로 받는 함수이다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20F%5By%28x%29%5D%20%3D%20%5Cint%20G%28y%2C%20x%29dx"/>
- 범함수 F 의 도함수는 다음과 같다. y 는 함수이지만 실수형 입력변수 처럼 생각하고 G 를 편미분했다는 의미로 partial 기호를 사용한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20F%7D%7B%5Cdelta%20y%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20y%7D"/>

#### ```기댓값 범함수의 도함수```
- 확률밀도함수 p(x) 를 입력변수로 받는 기댓값 E 는 다음과 같이 정의되는 범함수 이다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20E%5Bp%28x%29%5D%20%3D%20%5Cint%20xp%28x%29dx"/>
- 범함수의 정의에서 G(y, x) 에 기댓값 정의를 대입하면,
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20G%28y%2C%20x%29%20%3D%20xy"/> 이다.
- 따라서 y(x)=p(x) 에 대한 E 의 도함수는
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20F%7D%7B%5Cdelta%20y%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20y%7D%20%3D%20x"/>

#### ```그래디언트 부스팅의 도함수```
- 그래디언트 부스팅 gradient boosting 방법은 주어진 목표함수 y(x) 와 가장 비슷한 모형함수 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20y%28x%29"/> 를 구하기 위해 범함수의 손실함수를 사용하는 것을 말한다.
    - 최적화 optimization : 목적함수가 주어졌을 때 목적함수의 값을 최대, 최소로 만드는 모수를 찾는 것
    - 목적함수 : 최적화의 대상이 되는 함수. 성능의 크기를 측정하는 성능함수 performance function, 모형의 오차나 오류를 측정하는 손실함수 loss function, 오류함수 error function, 비용함수 cost function 등이 있다.
    - PCA 의 의미는 원래 차원의 행렬 x 와 유사한 행렬 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20%7B%5Chat%20x%7D"/> 를 찾는 문제이다. 이를 위해 차원축소한 행렬에 역변환행렬을 곱하여 원래 차원으로 만든 행렬과 원래 차원 행렬 x 와의 차를 최소화하는 가장 차이가 적은 유사한, 차원축소행렬을 찾는 과정을 거친다.
    >- <img src="https://latex.codecogs.com/gif.latex?arg%5C%3Bmin_%7B%5Chat%20x%7D%20%5C%7C%20x-U%5Chat%20x%20%5C%7C%5E2%2C%5C%3B%5C%3BU%5Chat%20x%20%3D%20%5Chat%7B%5Chat%20x%7D"/>
    - 이 최적화 과정에서 최적화의 대상이 된 함수를 손실함수라고 할 수 있다.
- 범함수의 손실함수
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20L%20%3D%20%5Cint%20%5Cdfrac%7B1%7D%7B2%7D%28%5Chat%20y%28x%29%20-%20y%28x%29%29%5E2%20dx"/>
>>- <img src="https://latex.codecogs.com/gif.latex?G%28%5Chat%20y%29%3D%5Cdfrac%7B1%7D%7B2%7D%28%5Chat%20y%28x%29%20-%20y%28x%29%29%5E2"/> (풀어서 정리한 후 미분)
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20%5Chat%20y%7D%20%3D%20%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20%5Chat%20y%7D%20%3D%20%5Chat%20y%28x%29%20-%20y%28x%29"/>

#### ```오일러-라그랑주 공식```
- x, y(x) 입력변수 이외에  추가로 y(x) 의 x 에 대한 도함수인 <img src="https://latex.codecogs.com/gif.latex?y%5E%7B%5Cprime%7D%28x%29%3D%5Cdfrac%7Bdy%7D%7Bdx%7D"/> 를 입력 변수로 받는 함수 G 의 도함수도 있을 수 있다.
    - G(y(x), y'(x), x)
>- <img src="https://latex.codecogs.com/gif.latex?%5Ctext%20F%5By%28x%29%5D%20%3D%20%5Cint%20G%28y%2C%20y%5E%7B%5Cprime%7D%2C%20x%29dx"/>
- 이러한 범함수 F 의 함수 y 에 대한 도함수
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%20%7B%5Cpartial%20F%7D%7B%5Cpartial%20y%7D%20-%20%5Cdfrac%7Bd%7D%7Bdx%7D%20%5Cleft%28%20%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20y%27%7D%5Cright%29"/>
- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%20%7B%5Cpartial%20F%7D%7B%5Cpartial%20y%7D"/> 와 <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20y%27%7D"/> 는 함수 y와 y'을 별개의 변수처럼 생각하고 편미분 한 것을 의미한다. <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%20%5Cleft%28%20%5Cdfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20y%27%7D%5Cright%29"/> 는 편미분하여 구해진 함수를 다시 변수 x 로 미분한 결과를 의미한다.
- 즉 G 를 y 와 y 의 도함수로 각각 편미분하고, y' 으로 편미분한 것을 다시 x 로 미분한다.

#### ```최적제어 optional control```
- 최적화는 어떤 함수 f(x) 의 값을 최대 또는 최소로 만드는 독립변수 x 의 값을 찾는 것이다.
- 최적제어는 범함수 F[f(x)] 의 값을 최대 또는 최소로 만드는 독립함수 y(x) 를 찾는 것이다.
- 최적화의 필요조건은 도함수 값이 0 이 되는 x 값이 어야 한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%28x%5E*%29%3D0"/>
- 최적제어도 마찬가지로 범함수의 도함수 값이 0 이 되는 최적의 함수 y(x) 이어야 한다.
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20f%7D%7B%5Cdelta%20y%7D%5By%5E*%28x%29%5D%3D0"/>
- ```GAN general adversarial network``` : 딥러닝에서 현실 데이터와 가장 유사한 데이터를 재현하는 방법
    - 두 확률분포함수 p_data 와 p_model 이 있을 때 범함수의 값을 최대화하는 확률분포함수 p(x) 를 구한다.
>- <img src="https://latex.codecogs.com/gif.latex?L%5Bp%5D%20%3D%20%5Cdfrac%7B1%7D%7B2%7D%20%5Cint%20%28%5Clog%20%28p%28x%29p_%7Bdata%7D%28x%29%20&plus;%20%5Clog%20%281-p%28x%29%29p_%7Bmodel%7D%28x%29%29%20dx"/>
>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20p%7D%20%3D%20%5Cdfrac%7B1%7D%7B2%7D%20%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20p%7D%20%5Clog%20p%28x%29p_%7Bdata%7D%28x%29%20&plus;%20%5Cdfrac%7B1%7D%7B2%7D%20%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20p%7D%20%5Clog%20%281-p%28x%29%29p_%7Bmodel%7D%28x%29"/>
>- 최적제어에 의한 최적의 확률분포 함수 p*(x) 는 이 값이 0 이 되도록 하는 값이다.
>>- <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%7Bp_%7Bdata%7D%28x%29%281-p%28x%29%29%20-%20p%28x%29p_%7Bmodel%7D%28x%29%7D%7B2p%28x%29%281-p%28x%29%29%7D"/>
>>- <img src="https://latex.codecogs.com/gif.latex?p%5E*%28x%29%20%3D%20%5Cdfrac%7Bp_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29%20&plus;%20p_%7Bmodel%7D%28x%29%7D"/>

