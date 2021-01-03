# Advanced Linear Algebra

### 1. 선형대수와 해석기하

#### 벡터의 기하학적 의미
- 벡터 a 는 n 차원 공간에서 점 또는 원점에서 점까지의 화살표를 의미한다.
- n 차원 공간에서의 벡터 a 는 평행이동 할 수 있다.
- 벡터의 길이는 놈 norm 으로 정의한다.
    - ```벡터의 길이``` : <img src="https://latex.codecogs.com/gif.latex?%5C%7Ca%5C%7C%3D%5Csqrt%7Ba%5ETa%7D%3D%5Csqrt%7Ba_%7B1%7D%5E2&plus;a_%7B2%7D%5E2&plus;a_%7B3%7D%5E2&plus;...&plus;a_%7BN%7D%5E2%7D"/>
    - 원래 놈의 제곱은 이런 형태이다. <img src="https://latex.codecogs.com/gif.latex?%5C%7Ca%5C%7C%5E2%3Da%5ETa%3D%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5ENa_%7Bi%2Cj%7D%5E2"/>

#### 스칼라와 벡터의 곱
- 양의 실수 c 와 벡터 a 를 곱하면 방향은 고정, 길이가 실수 크기만큼 커진다.
- 음의 실수 c 와 벡터 a 를 곱하면 방향은 반대가 되고, 길이가 실수 크기만큼 커진다.

#### 단위벡터 unit vector
- ```길이가 1 인 벡터를 단위벡터라고 한다.``` (놈의 정의에 의해서 원소들의 제곱합이 1 인 벡터들은 모두 단위벡터 이다.)
- 영벡터가 아닌 어떤 벡터 x 에 대하여 x 의 놈, 즉 길이로 나눈 것은 x 의 단위벡터가 된다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdfrac%20x%7B%5C%7Cx%5C%7C%7D"/>

#### 벡터의 합
- 두 벡터의 합은 두 벡터를 이웃하는 변으로 가지는 평행사변형의 대각선 벡터가 된다.

#### 벡터의 선형조합
- 여러개의 벡터에 스칼라곱을 한 후 모두 더한 것을 ```선형조합 linear combination``` 이라고 한다.
- n 차원 공간에서 벡터의 선형조합의 의미는, 각 벡터에 가중치가 곱해져 길이나 방향이 변하고, 변한 벡터들을 더한 위치의 벡터가 된다.

#### 벡터의 차
- a-b=c 는 벡터 b 가 가리키는 점으로부터 벡터 a 가 가리키는 점을 연결하는 벡터다.
- b + c = b + (a - b) = a

#### wrod2Vec (인공신경망)
- 벡터의 차를 활용한 단어의미 분석
- 단어의 의미를 벡터로 표현하고, 벡터의 평행이동을 적용하여 다른 단어에도 적용함으로써 같은 의미를 찾을 수 있다.
    - ```한국 - 서울```은 ```서울->한국```으로 향하는 벡터일 때, 이것은 수도이름을 나라이름으로 바꾸는 행위와 같다.
    - 이러한 행위를 파리에 적용하면 ```파리 + (한국-서울)``` 이라고 벡터로 표시 할 수 있다.
    - 이러한 벡터의 연산을 word2Vec 에 학습 시키면, 파리와 가장 가까이에 위치한 단어인 ```프랑스```가 나온다.

#### 유클리드 거리
- ```유클리드 거리 Euclidean distance``` : 두 벡터가 가리키는 점 사이의 거리
- 벡터의 차의 길이로 구한다.
- <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Ctiny%20%5C%7Ca-b%5C%7C%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%28a_i-b_i%29%5E2%7D%3D%5Csqrt%7B%5Csum_%7Bi%3D1%7D%20%28%20a_i%5E2%20-%202%20a_i%20b_i%20&plus;%20b_i%5E2%20%29%7D%3D%5Csqrt%7B%5Csum_%7Bi%3D1%7D%20a_i%5E2%20&plus;%20%5Csum_%7Bi%3D1%7D%20b_i%5E2%20-%202%20%5Csum_%7Bi%3D1%7D%20a_i%20b_i%7D%3D%20%5Csqrt%7B%5C%7C%20a%20%5C%7C%5E2%20&plus;%20%5C%7C%20b%20%5C%7C%5E2%20-%202%20a%5ETb%20%7D"/>
- <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Ca-b%5C%7C%5E2%20%3D%20%5C%7Ca%5C%7C%5E2&plus;%5C%7Cb%5C%7C%5E2-2a%5ETb"/>

#### 벡터의 내적과 삼각함수
- 두 벡터의 내적은 벡터의 길이와 벡터사이의 각도의 ```코사인 함숫값```으로 계산할 수 있다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5ETb%3D%5C%7Ca%5C%7C%5C%7Cb%5C%7Ccos%5Ctheta"/>
    - 삼각함수 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20sin%5Ctheta%24%2C%20%24cos%5Ctheta%24%2C%20%24tan%5Ctheta"/>
    - 빗변과 높이의 비율 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20sin%5Ctheta%3D%5Cdfrac%7Ba%7D%7Bh%7D"/>, 빗변과 밑변의 비율 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20cos%5Ctheta%3D%5Cdfrac%7Bb%7D%7Bh%7D"/> (따라서 각도에 따라 반대의 성질을 갖는다.)
    - 두 벡터의 각도가 90도 일때 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20sin%5Ctheta%3D1%24%2C%20%24cos%5Ctheta%3D0"/>
    - 두 벡터의 각도가 0도 일때 (방향이 완전히 같을때) : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20sin%5Ctheta%3D0%24%2C%20%24cos%5Ctheta%3D1"/>

#### 직교
- ```직교 orthgonal``` : 두 벡터 사이의 각도가 90도 일때 직교한다고 정의한다.
- 각도가 90도 이면 $cos\theta=0$ 이므로, 두 벡터의 내적은 공식에 적용하면, 0 이 된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5ETb%20%3D%20b%5ETa%20%3D%200%20%5C%3B%5C%3B%5C%3B%20%5Cleftrightarrow%20%5C%3B%5C%3B%5C%3B%20a%20%5Cperp%20b"/>
    - 내적값이 0 이면 직교한다고 할 수 있다.
- 행렬은 기본적으로 각 열벡터가 기저를 이루는 좌표계 cordinate system 이다.
    - 직교행렬 : 행렬의 모든 열벡터들이 서로 직교
    - 정규직교행렬 : 행렬이 직교행렬이고 모든 열벡터의 크기가 1이면 정규직교
- 어떤 행렬이 직교행렬인지 아닌지 확인하려면, 열벡터들의 내적이 0인지 아닌지 확인하면 된다.

#### 정규직교
- ```정규직교 orthonormal``` : ```N 개의 단위벡터``` v1,v2,v3,...vN 이 서로 직교하면 정규직교라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Cv_%7Bi%7D%5C%7C%20%3D%201%20%5C%3B%5C%3B%20%5Cleftrightarrow%20%5C%3B%5C%3B%20v_%7Bi%7D%5ETv_%7Bj%7D%20%3D%201"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_%7Bi%7D%5ETv%7Bj%7D%20%3D%200%20%5C%3B%5C%3B%20%28i%20%5Cneq%20j%29%20%24%2C%20%24v_%7Bi%7D%5ETv_%7Bj%7D%20%3D%201%20%5C%3B%5C%3B%20%28i%20%3D%20j%29"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%5ETV%3DI%24%2C%20%24V%5E%7B-1%7D%3DV%5E%7BT%7D"/>

#### 코사인 유사도
- ```코사인 유사도 cosine similarity``` : 두 벡터의 방향이 비슷할 수록 벡터가 비슷하다고 간주하여, 두 벡터 사이의 각의 코사인 값을 말한다.
- 각도가 0 일때 코사인값이 가장 커지므로, 두 벡터가 같은 방향을 가리키면 코사인 유사도가 최댓값 1을 가진다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctext%20cosine%20%5C%3B%20similarity%20%3D%20%5Ccos%5Ctheta%3D%5Cdfrac%7Bx%5ETy%7D%7B%5C%7Cx%5C%7C%5C%7Cy%5C%7C%7D"/> (벡터의 내적 공식에서 도출)
- ```코사인 거리 cosine distance``` : 코사인 유사도를 사용하여 두 벡터간의 거리를 측정한다. ```추천시스템 recommender system``` 에서 사용된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctext%20cosine%20%5C%3B%20distance%20%3D%201-cosine%5C%3Bsimilarity%20%3D%201-%5Cdfrac%7Bx%5ETy%7D%7B%5C%7Cx%5C%7C%5C%7Cy%5C%7C%7D"/>
    - 분모는 벡터의 길이간의 곱이므로 내적이 아니다.

#### 벡터의 성분과 분해
- 벡터 a 와 b 의 합으로 벡터 c 를 만들 수 있다. 이때 벡터 c 는 ```성분 component``` a, b 로 ```분해 decomposition``` 된다고 한다.

#### 벡터의 투영성분과 직교성분
- 벡터 a 는 벡터 b 에 대한 ```투영성분 projection``` 과 ```직교성분 rejection``` 으로 분해 된다.
- 벡터 a 에 수직으로 햇빛이 내리쬐면 바닥에 생기는 그림자를 투영성분, 벡터 a 에서 바닥으로 그은 직선을 직교성분이라고 한다.
    - 투영성분 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5E%7B%5CVert%20b%7D"/>
    - 투영성분의 길이 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7C%20a%5E%7B%5CVert%20b%7D%20%5C%7C%20%3D%20%5C%7Ca%5C%7C%5Ccos%5Ctheta%20%3D%20%5Cdfrac%7B%5C%7Ca%5C%7C%5C%7Cb%5C%7C%5Ccos%5Ctheta%7D%7B%5C%7Cb%5C%7C%7D%20%3D%20%5Cdfrac%7Ba%5ETb%7D%7B%5C%7Cb%5C%7C%7D%20%3D%20%5Cdfrac%7Bb%5ETa%7D%7B%5C%7Cb%5C%7C%7D%20%3D%20a%5ET%5Cdfrac%7Bb%7D%7B%5C%7Cb%5C%7C%7D"/>
    - 투영성분 그 자체 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5E%7B%5CVert%20b%7D%20%3D%20%5Cdfrac%7Ba%5ETb%7D%7B%5C%7Cb%5C%7C%7D%20%5Cdfrac%7Bb%7D%7B%5C%7Cb%5C%7C%7D%3D%20%5Cdfrac%7Ba%5ETb%7D%7B%5C%7Cb%5C%7C%5E2%7Db%24%20%5C%3B%20%28%24%5Cdfrac%7Bb%7D%7B%5C%7Cb%5C%7C%7D%29"/> 는 벡터 b 의 단위벡터이다.)
    - b 가 단위 벡터 일 경우 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7C%20a%5E%7B%5CVert%20b%7D%20%5C%7C%20%3D%20a%5ETb"/> ; (단위벡터의 길이가 1이므로, a, b 의 내적이 된다.)
    - 직교성분 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5E%7B%5Cperp%20b%7D"/>
    - 직교성분은 투영성분을 뺸 나머지이다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20a%5E%7B%5Cperp%20b%7D%20%3D%20a%20-%20a%5E%7B%5CVert%20b%7D"/>

#### 직선의 방정식
- 원점에서 출발하는 벡터 w 가 가리키는 지점을 지나고, 벡터 w 에 직교하는 어떤 직선 A.
- ```직선 A 위의 임의의 점 x 와 벡터 w 사이의 벡터.```
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20w%5ET%28x-w%29%20%3D%200%24%2C%20%24w%5ETx%20-%20w%5ETw%20%3D%20w%5ETx%20-%20%5C%7Cw%5C%7C%5E2%20%3D%200"/>,   (벡터 w 와 벡터 x-w 가 직교하므로)
    - 직선 A 와 원점 사이의 거리 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Cw%5C%7C"/>  (벡터 w 의 길이)
- ```벡터 w 의 점을 지나지 않고, 벡터 w 에 직교인 직선 A 의 방정식```
    - 직선 A 는 벡터 w 의 점을 지나지는 않지만, 벡터 w 위의 어떤 점 cw 는 지난다. (c 는 임의의 상수)
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20cw%5ETx%20-%20c%5E2%5C%7Cw%5C%7C%5E2%20%3D%20w%5ETx%20-%20c%5C%7Cw%5C%7C%5E2%20%3D%200"/>
    - cw 가 임의의 점이므로 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20w%5ETx%20-%20w_0%20%3D%200"/>
    - 직선 A 와 원점 사이의 거리 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20c%5C%7Cw%5C%7C%20%3D%20%5Cdfrac%7Bw_0%7D%7B%5C%7Cw%5C%7C%7D"/>

#### 직선과 점의 거리
- 직선 A 가 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20w%5ETx%20-%20%5C%7Cw%5C%7C%5E2%20%3D%200"/> 이고, 이 위에 있지 않은 임의의 점 x' 와의 거리
- x' 는 벡터 w 에 대해서 투영성분으로 분해 할 수 있다. 이 x' 의 벡터 w 에대한 투영성분으로 직선 A 와의 거리를 구한다.
    - 투영성분의 길이 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Cx%5E%7B%5CVert%20w%7D%5C%7C%20%3D%20%5Cfrac%7Bw%5ETx%27%7D%7B%5C%7Cw%5C%7C%7D"/>
    - 점 x' 와 직선 A 의 거리는 벡터 w 와 벡터 x' 의 투영성분의 차와 같다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cleft%7C%20%5C%7Cx%27%5E%7B%5CVert%20w%7D%5C%7C%20-%20%5C%7Cw%5C%7C%20%5Cright%7C%20%3D%20%5Cleft%7C%20%5Cdfrac%7Bw%5ETx%27%7D%7B%5C%7Cw%5C%7C%7D%20-%20%5C%7Cw%5C%7C%20%5Cright%7C%20%3D%20%5Cdfrac%7B%5Cleft%7Cw%5ETx%27%20-%20%5C%7Cw%5C%7C%5E2%20%5Cright%7C%7D%7B%5C%7Cw%5C%7C%7D"/>
- 직선 A 가 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20w%5ETx%20-%20%5C%7Cw_0%5C%7C%20%3D%200"/> 이고, 이 위에 있지 않은 임의의 점 x' 와의 거리
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cleft%7C%20%5C%7Cx%27%5E%7B%5CVert%20w%7D%5C%7C%20-%20c%5C%7Cw%5C%7C%20%5Cright%7C%20%3D%20%5Cdfrac%7B%5Cleft%7Cw%5ETx%27%20-%20w_0%20%5Cright%7C%7D%7B%5C%7Cw%5C%7C%7D"/>, (<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20w_0%20%3D%20c%5C%7Cw%5C%7C%5E2"/>)
- 직선과 점사이의 거리는 분류방법의 하나인 ```서포트 벡터 머신 SVM, support vector machine``` 에서 사용된다.

### 2. 좌표와 변환

#### 선형종속과 선형독립
- ```선형종속 linearly dependent``` : 선형조합이 0이 되게끔 만드는 계수가 모두 0이 아닌 경우가 존재하는 경우. 즉 계수 c 가 0이 아니어도 선형조합의 값이 0이 되는 경우.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20c_%7B1%7Dx_%7B1%7D%20&plus;%20c_%7B2%7Dx_%7B2%7D%20&plus;%20...%20&plus;%20c_%7BN%7Dx_%7BN%7D%20%3D%200"/>
- ```선형독립 linearly independent``` : 선형조합이 0이 되게끔 만드는 계수가 모두 0이어야만 하는 경우. 즉 계수 c 가 0이 아니면 선형조합의 값이 0이 안되는 경우.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20c_1%20x_1%20&plus;%20%5Ccdots%20&plus;%20c_N%20x_N%20%3D%200%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20c_1%20%3D%20%5Ccdots%20%3D%20c_N%20%3D%200"/> (반드시 계수 모두가 0 이어야 한다.)
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20c_1%20x_1%20&plus;%20%5Ccdots%20&plus;%20c_N%20x_N%20%3D%200%20%5C%3B%5C%3B%20%5Cleftrightarrow%20%5C%3B%5C%3B%20c_1%20%3D%20%5Ccdots%20%3D%20c_N%20%3D%200"/> (계수 모두가 0 이면 선형조합이 0 이다라는 조건 추가)
- 모든 벡터들간에 선형독립이 성립하지 않는 경우 : 3 개의 2차원 벡터들, 4 개의 3차원 벡터들. 미지수의 갯수가 방정식의 갯수보다 많으므로 해의 갯수가 무한대이다. 즉 계수들이 0 이 아니어도 선형조합의 값이 0이 될 수 있는 경우가 많다.
- 선형독립인 벡터를 찾을 때 : 벡터의 요소별 비율이 같으면 선형독립이 된다.

#### 선형독립과 선형 연립방정식
- 선형독립 관계를 행렬의 선형연립방정식 형태로 나타낼 수 있다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20c_%7B1%7Dx_%7B1%7D&plus;c_%7B2%7Dx_%7B2%7D&plus;...&plus;c_%7BN%7Dx_%7BN%7D%20%3D%20Xc"/>
    - 벡터의 선형독립 문제는 선형연립방정식을 푸는 것과 같다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20Xc%20%3D%200"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20Xc%20%5C%3B%5C%3B%20%5Crightarrow%20%5C%3B%5C%3B%20c%20%3D%200%24%2C%20%24Xc%20%5C%3B%5C%3B%20%5Cleftrightarrow%20%5C%3B%5C%3B%20c%20%3D%200"/>

#### 선형종속의 대표적인 예
- 데이터의 열벡터들이 선형종속이면 다중공선성 multicollinearity 라고 한다. 예측 성능이 떨어진다. 안 좋은 데이터.
- 벡터가 선형종속이 되는 대표적인 경우
    - ```벡터의 개수가 벡터의 차원보다 크면 선형종속이다.``` : 열이 행보다 많다. 특징이 데이터보다 많다. 미지수가 방정식보다 많다. 해가 많다.
    - ```중복 데이터가 있으면 반드시 선형종속이다.``` : i, j 번째 열벡터가 같으면 선형종속이다. c 가 0 이 아닌 경우에도 선형조합이 0 이 된다.
    - ```어떤 벡터가 다른 벡터의 선형조합이면 선형종속이다.``` : 주차별 매출이 각각 다른 벡터인데, 매출 평균이 다른 벡터에 들어 있는 경우.

#### 랭크
- 랭크 rank : 어떤 행렬 A 에서 서로 선형독립인 벡터들의 최대 갯수. 스칼라.
    - ```열랭크 column rank``` : 열벡터 간의 선형독립인 열벡터들의 최대 갯수
    - ```행랭크 row randk``` : 행벡터 간의 선형독립인 행벡터들의 최대 갯수
- 랭크의 성질
    - ```행랭크와 열랭크는 같다.``` 즉 행 기준으로 선형조합을 따지든, 열 기준으로 선형조합을 따지든 선형독립 벡터의 최대갯수는 같다.
    - 행랭크는 행의 갯수보다 커질 수 없고, 열벡터는 열의 갯수보다 커질 수 없다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20rankA%20%5Cleq%20min%28M%2CN%29"/> (M,N 중 작은 것과 같거나 작다.)
    - 랭크가 1인 경우도 있다.
- ```풀랭크 full rank``` : 랭크가 행이나 열의 갯수 중 작은 값과 같으면 풀랭크라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctext%20rankA%20%3D%20min%28M%2CN%29"/> (M,N 중 작은 것과 같다.)
    - 벡터간의 선형조합의 값이 0 인 계수 c1,c2...cn 이 모두 0 일 때만 가능한 경우 ```선형독립```이라고 하며, 어떤 행렬에서 선형독립인 벡터들의 최대 갯수를 ```랭크```라고 한다. ```풀랭크```는 이러한 랭크의 갯수가 행과 열의 갯수 중 작은 값과 같은 경우를 말한다.
    - 선형독립인 벡터들로 행렬을 만들었을 때 항상 풀랭크이다.
    - 풀랭크 인 행렬일 수록 좋은 데이터이다.
    - 위의 성질에 따라서 4x3 행렬에서 랭크는 3을 넘을 수 없고, 행과 열의 랭크가 같으므로, 3개의 열에서 랭크의 갯수를 찾는 것이 효율적이다. 랭크가 2가 나오면 풀랭크가 아니다. 최소 3개가 나와야 풀랭크 행렬이다.
- ```로우 랭크 행렬 low rank matrix```
    - N 차원 벡터 x 한 개로 만들어지는 정방 행렬을 ```랭크-1 행렬``` 이라고 한다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20xx%5ET%20%5Cin%5Cmathbf%20R%5E%7BN%5Ctimes%20N%7D"/>
    - 랭크-1 행렬은 고윳값과 고유벡터의 정리에서 사용된다.
    - 벡터 x 는 기본적으로 열벡터이므로 열벡터와 행벡터의 곱의 형태이므로 정방행렬이 된다.
        - 랭크-1 행렬의 랭크는 1이다. 열벡터, 행벡터가 곱해져서 정방행렬로 뻥튀기 된 것. 쓸모 있는 것은 하나밖에 없다.
    - N 차원 벡터 x 두 개로 만들어지는 다음 행렬을 ```랭크-2 행렬``` 이라고 한다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_1x_1%5ET%20&plus;%20x_2x_2%5ET"/>
        - 랭크-2 행렬의 랭크는 2이다.
    - N 차원 벡터 x M 개로 만들어지는 다음 행렬을 ```랭크-M 행렬``` 이라고 한다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_1x_1%5ET%20&plus;%20x_2x_2%5ET%20&plus;%20...%20&plus;%20x_Mx_M%5ET%20%3D%20%5Csum_%7Bi%3D1%7D%5EMx_ix_i%5ET"/>
        - 랭크-M 행렬은 ```특잇값 분해와 PCA principal component analysis 에서 사용 된다.```
- ```정방행렬 X 가 풀랭크 이면 역행렬이 존재한다.``` 풀랭크의 여부로 역행렬이 있는지 없는지 확인 할 수 있다.
    - 정방행렬이 풀랭크 이면, 행과 열 벡터 모두가 선형독립이다.
    - 역행렬이 존재한다면, 선형회귀모델에서 가중치 x 의 값을 구할 수 있다. <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%3DA%5E%7B-1%7Db"/>

#### 벡터공간과 기저벡터
- ```벡터공간 vector space``` : 서로 선형독립인 벡터 N 개를 선형조합하여 만들어지는 벡터의 집합을 벡터공간이라고 한다. $V$
    - 벡터공간의 차원 : 벡터공간의 차원은 벡터공간을 이루는 벡터의 갯수 N 개 (벡터의 차원은 벡터의 원소의 수)
    - 벡터 100개를 선형조합하여 만든 벡터공간 V 의 차원은 100차원이다.
    - 서로 선형독립인 벡터 N 를 벡터공간의 ```기저벡터 basis vector``` 라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%20%3D%20%5C%7Bc_1x_1%20&plus;%20%5Ccdots%20&plus;%20c_Nx_N%20%5C%3B%20%5Cvert%20%5C%3B%20c_1%2C%20%5Cldots%2C%20c_N%20%5Cin%5Cmathbf%20%7BR%7D%20%5C%7D"/>
- N 차원 벡터 N 개가 선형독립이면 아래의 정리가 성립한다.
    - N 개의 N 차원 벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_1%2Cx_2%2C%5Ccdots%2Cx_N"/> 이 선형독립이면, 이를 선형조합하여 모든 N 차원 벡터를 만들 수 있다.
    - x_1=np.array([[1],[2]]), x_2=np.array([[2],[1]]) 두 벡터는 서로 선형독립이다. 이 정리에 따라서 두 벡터를 계수 c_1, c_2 와 선형조합하여 어떠한 2 차원 벡터도 만들 수 있다. 이러한 선형조합하여 만들어진 벡터들의 집합을 벡터공간이라고 한다.
    - 선형독립이 아닌 벡터는 벡터공간의 기저벡터가 될 수 없다.
    - x_1=np.array([[1],[2],[0]]), x_2=np.array([[2],[1],[0]]) 은 선형독립이며 벡터공간의 기저벡터이다. 이 벡터공간은 2개의 벡터로 만들어졌으므로 2차원이다. (3차원이 아니다.)
- 벡터공간의 차원과 벡터의 차원의 기준이 다른 이유는 기저벡터를 선형조합하여 만들지 못하는 벡터들이 있기 때문.

#### 랭크와 역행렬
- 정방행렬의 랭크와 역행렬 사이의 정리.
    - ```정방행렬이 풀랭크면 역행렬이 존재한다. 역도 성립한다. 즉, 정방행렬의 역행렬이 존재하면 풀랭크이다.```
    - 정방행렬이 풀랭크이다. ↔ 역행렬이 존재한다.
    - -> 방향의 증명 : 기저벡터의 선형조합으로 항등행렬을 만들 수 있다고 가정하면, XC=CX=I 가 성립한다. 역행렬의 성질에 의해서 C 는 X 의 역행렬이 된다.
    - <- 방향의 증명 : 선형연립방정식과 선형독립의 관계에서 벡터 N 개가 선형독립 일때의 논리기호 Xc=0 ↔ c=0 를 사용한다. c=0 이면 Xc=0 성립, X의 역행렬이 존재한다고 가정하면 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20X%5E%7B-1%7DXc%3Dc%3D0"/> 이 성립한다. 따라서 X 의 역행렬이 존재한다.
- 어떤 정방행렬 데이터가 풀랭크이면 역행렬이 존재하고, 역행렬이 존재하면 선형조합의 해와 선형연립방정식의 최소자승문제의 해를 의사역행렬을 통해서 구할 수 있게 된다.

#### 벡터공간과 투영벡터, 직교벡터
- N 차원 벡터 M 개 v1, v2, ... , vm 으로 이루어진 기저벡터가 있을 때, N 차원 벡터 x 와 이 기저벡터 v1,v2,...,vm 을 선형조합하여 벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5E%7B%5CVert%20v%7D"/> 를 만들었다.
- 이 벡터와 벡터 x 의 차 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x-x%5E%7B%5CVert%20v%7D"/> 인 벡터 a가 모든 기저벡터 v1,v2,...,vm 에 대하여 직교할 때,
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x-x%5E%7B%5CVert%20v%7D%3Dx%5E%7B%5Cperp%20v%7D"/> 벡터를 ```벡터공간 V 에 대한 직교벡터```라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5E%7B%5CVert%20v%7D"/> 벡터를 ```벡터공간 V 에 대한 투영벡터```라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x-x%5E%7B%5CVert%20V%7D%20%5Cperp%20%5C%7Bv_1%2Cv_2%2C%5Ccdots%2Cv_M%5C%7D"/>
- v1,v2,...,vm 은 기저벡터이므로 이미 선형독립인 벡터들이다.
- M=2, N=3 으로 정의하면, 2차원 평면에 3차원 벡터가 투영 된 것을 확인 할 수 있다.

#### 정규직교인 기저벡터로 이루어진 벡터공간
- 기저벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_1%2Cv_1%2C...%2Cv_M"/> 이 정규직교이면, 투영벡터는 각 기저벡터에 대한 내적값으로 표현할 수 있다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5E%7B%5CVert%20V%7D%3D%28x%5ETv_1%29v_1&plus;%28x%5ETv_2%29v_2&plus;%5Ccdots&plus;%28x%5ETv_M%29v_M"/>
    - ```!!! 투영벡터가 내적의 결과라는 것은 알겠는데, 내적에서 v_1 이 왜 두번 곱해지는지 확인 할것, 이해 안됨.```
- 이러한 투영벡터의 길이의 제곱은 각 기저벡터와의 내적의 제곱합이다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Cx%5E%7B%5CVert%20V%7D%5C%7C%5E2%3D%5Csum_%7Bi%3D1%7D%5EM%28x%5ETv_i%29%5E2"/>, (위의 식 정리)
    - 벡터 x 에서 투영벡터를 뺴면 직교벡터가 된다. (증명 가능)
- 직교벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5E%7B%5Cperp%20V%7D"/> 는 기저벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_1%2C%5Ccdots%2Cv_M"/> 으로 이루어진 벡터공간의 모든 벡터에 대해서 직교한다.
- 따라서 벡터 x 의 투영벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5E%7B%5CVert%20V%7D"/> 는 기저벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_1%2C%5Ccdots%2Cv_M"/> 이루어진 벡터공간의 모든 벡터 중에서 벡터 x 와 가장 가까운 벡터이다.

#### 표준기저벡터
- 표준기저벡터 standard basis vector : 기저벡터 중에서 원소 중 하나의 값이 1 이고 나머지는 0 으로 이루어진 것.
    - 표준기저벡터들을 열벡터로 갖는 행렬은 항등행렬이 된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Be_1%2Ce_2%2C%5Ccdots%2Ce_N%5D%20%3D%20I_N"/>

####  좌표
- ```좌표 coordinate``` : 어떤 벡터 x 를 나타내기 위해 기저벡터를 선형조합하여 만든 계수벡터를 말한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%20%3D%20x_%7Be_1%7D%20e_1%20&plus;%20x_%7Be_2%7D%20e_2"/>
- 어떤 벡터 x 가 있을 때 이 벡터의 위치를 표시하는 것은 어떤 기저벡터를 기준으로 했느냐에 따라 달라진다. 이러한 기준이 되는 기저벡터로 벡터 x 의 위치를 표시한 것을 좌표라고 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%20%3D%20%5B%20e_1%20e_2%20%5D%20%5Cbegin%7Bbmatrix%7D%20x_%7Be_1%7D%20%5C%5C%20x_%7Be_2%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5B%20e_1%20e_2%20%5D%20%5C%3B%20x_e"/> (벡터 x 를 기저벡터로 나타낸 것)
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_e%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_%7Be_1%7D%20%5C%5C%20x_%7Be_2%7D%20%5Cend%7Bbmatrix%7D"/> (x_e 가 벡터 x 에 대한 기저벡터의 좌표가 된다.)
- 기저벡터가 바뀌면 벡터 x 는 그대로 이지만 좌표는 새 기저벡터에 맞게 바뀐다. 즉 기존의 기저벡터 e 에서의 좌표와 새로운 기저벡터 g 에서의 좌표는 다르다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20g_1%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%200%20%5Cend%7Bbmatrix%7D%2C%20%5C%3B%20g_2%20%3D%20%5Cbegin%7Bbmatrix%7D%20-1%20%5C%5C%201%20%5Cend%7Bbmatrix%7D"/> (새로운 기저벡터)
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_g%20%3D%20%5Cbegin%7Bbmatrix%7D%204%20%5C%5C%202%20%5Cend%7Bbmatrix%7D"/> (벡터 x 에 대한 새로운 기저벡터의 좌표)
    - 이 의미는 벡터 x 는 g_1 방향으로 4 만큼, g_2 방향으로 2 만큼 이동한 벡터의 합이 가리키는 지점이라는 뜻이다.
- 그러나 같은 벡터 x 에 대한 상대적인 값이기 때문에 기저벡터간의 호환할 수 있는 기준이 필요하다. 즉 기존 기저벡터와 새로운 기저벡터를 서로를 사용해서 정의할 수 있어야 한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%20g_1%20%26%20g_2%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20e_1%20%26%20e_2%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20g_%7B1e%7D%20%26%20g_%7B2e%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20e_1%20%26%20e_2%20%5Cend%7Bbmatrix%7D%20A"/>
    - g1e, g2e 를 열벡터로 묶은 행렬 A
- ```변환행렬 transform matrix``` : 벡터 x 를 나타내는데 기존의 기저벡터와 새로운 기저벡터를 서로 호환하기 위한 행렬 A 의 역행렬 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%5E%7B-1%7D%3DT"/>
- ```좌표변환 coordinate transform``` : 새로운 기저벡터에 대해 좌표를 계산하는 것
- 벡터 x 의 기저벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Be_1%2C%20e_2%5C%7D"/> 의 좌표 xe 를 새로운 기저벡터 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5C%7Bg_1%2Cg_2%5C%7D"/> 에 대한 좌표 xg 로 변환하면서 변환행렬이 정의 된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%20%3D%20x_%7Be1%7De_1%20&plus;%20x_%7Be2%7De_2%20%3D%20x_%7Bg1%7D%20g_1%20&plus;%20x_%7Bg2%7D%20g_2"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%20e_1%20%26%20e_2%20%5Cend%7Bbmatrix%7D%20x_e%20%3D%20%5Cbegin%7Bbmatrix%7D%20g_1%20%26%20g_2%20%5Cend%7Bbmatrix%7Dx_g"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%20g_1%20%26%20g_2%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20e_1%20%26%20e_2%20%5Cend%7Bbmatrix%7D"/> 을 대입한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x_e%20%3D%20A%20x_g%2C%20%5C%3B%5C%3B%20x_g%20%3D%20A%5E%7B-1%7Dx_e%20%3D%20Tx_e"/>

#### 이미지변환
- 좌표의 변환으로 이미지를 변환 할 수 있다.
    - 회전 : 기준이 되는 기저벡터가 바뀌면서 이미지의 방향이 바뀐다.
    - 스케일 : 기존 좌표에서 변환 된 좌표와의 비례에 따라서 이미지가 늘어나거나 줄어든다.
- 원점 자체를 바꿀 수 는 없다.
- scipy 패키지의 기저벡터 : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20e_1%3D%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%20-1%20%5Cend%7Bbmatrix%7D%24%2C%20%24e_2%3D%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%200%20%5Cend%7Bbmatrix%7D"/>

### 3. 고윳값분해
- 고윳값분해와 특잇값분해는 행렬의 내부구조를 살펴보거나, 행렬의 연산을 효율적으로 하기위해 유용하게 사용된다.
- 행렬의 좌표변환의 일종이다. 변환해서 방향은 안 바뀌고 길이에만 변화가 있는 벡터.

#### 고윳값과 고유벡터
- ```고유벡터 eigenvector``` : 정방행렬 A 를 곱해서 변환하려고 해도 변환되지 않는 벡터. 영벡터가 아니어야한다. <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v"/>
- ```고윳값 eigenvalue``` : 고유벡터의 변형 전,후의 크기의 비율. <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clambda"/>
- ```고윳값분해 eigenvalue decomposition, 고유분해 eigen-decomposition``` : 정방행렬 A 에서 고유벡터와 고윳값을 찾는 행위
- 행렬 A 에 대하여 고윳값과 고유벡터는 다음 식을 만족한다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20Av%20%3D%5Clambda%20v"/>
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%28A-%5Clambda%20I%29v%3D0"/>
- 어떤 벡터 v 가 고유벡터이면, 이 벡터에 실수를 곱한 모든 cv 벡터, v와 방향이 같은 벡터는 모두 고유벡터가 된다.
    - <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%28cv%29%3DcAv%3Dc%20%5Clambda%20v%3D%5Clambda%28cv%29"/>
    - 고유벡터는 길이가 1인 단위벡터로 정규화 하여 표시한다. : <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cdfrac%7Bv%7D%7B%5C%7Cv%5C%7C%7D"/>

#### 특성방정식
- 정방행렬 A 만 주어졌을 때 고윳값과 고유벡터를 구하는 방법.
- 고윳값은 <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A-%5Clambda%20I"/> 의 행렬식의 값을 0으로 만드는 특성방정식 characteristic equation 의 해와 같다.
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20det%28A-%5Clambda%20I%29%3D0
    - 이 조건에 의해서 https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A-%5Clambda%20I 는 역행렬을 갖지 않는다는 것을 알 수 있다. (역행렬 공식 확인)
    - 특성방정식의 해를 구하면 고윳값을 구할 수 있고, 행렬과 고윳값을 알고 있기 때문에 고유벡터도 구할 수 있게 된다.
    - 고윳값은 특성방정식의 해에 따라서 1개 (중복고윳값), 2개, 또는 실수가 아닌 복소수인 경우 로 달라지게 된다.

#### 고윳값의 갯수
- ```중복된 고윳값을 가진 경우 각각으로 생각하고, 복소수인 고윳값도 가능하다면, n 차원 정방행렬의 고윳값은 항상 n 개이다.```

#### 고윳값과 대각합/행렬식
- 대각합과 행렬식은 벡터와 행렬의 크기에 해당하는 개념들.
- 어떤 행렬의 고윳값이 https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clambda_1%2C%20%5Clambda_2%2C%20%5Ccdots%2C%20%5Clambda_n 라고 할 떄, 모든 고윳값의 합은 행렬의 대각합과 같고, 모든 고윳값의 곱은 행렬의 행렬식 값과 같다.
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20tr%28A%29%3D%5Csum_%7Bi%3D1%7D%5EN%20%5Clambda_i
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20det%28A%29%3D%5Cprod_%7Bi%3D1%7D%5EN%20%5Clambda_i
    - ```역행렬의 존재여부는 고윳값 중에 0 이 있느냐 없느냐로 판단 가능하다. 고윳값 중에 0 이 있으면, det(A) 가 0이 된다.```

#### 고유벡터의 계산
- 고윳값을 알면 다음 연립방정식을 풀어 고유벡터를 구할 수 있다.
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%28A-%5Clambda%20I%29v%3D0
    - 고윳값 각각 고유벡터를 구해야 한다.
    - 고윳값이 1개일때 고유벡터도 1개이거나 경우에 따라서 여러개 일 수 있다. v_11 = 2/3v_12 를 만족하는 모든 벡터 가 될 수도 있다.
    - 항등행렬 I 의 고윳값은 1 하나이지만, 고유벡터는 임의의 2차원 벡터 모두가 될 수 있다.
    - 단, 단위벡터는 1개 이다.
    - 고유벡터는 단위벡터로 정규화하여 나타내주는 것이 좋다.

#### 대각화
- ```대각화 diagonalization``` : N 차원 정방행렬을 인수분해 하는 행위.
    - 고유벡터행렬 https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V : 고유벡터를 열벡터로 옆으로 쌓아서 만든 행렬, https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%20%5Cin%5Cmathbf%20%7BR%7D%5E%7BN%5Ctimes%20N%7D, 정방행렬
    - 고유값행렬 https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5CLambda : 고윳값을 대각성분으로 가지는 대각행렬, https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5CLambda%20%5Cin%5Cmathbf%20%7BR%7D%5E%7BN%5Ctimes%20N%7D, 정방행렬
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20AV%3D%5CLambda%20V
    - V 의 역행렬이 존재한다면 : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%3DV%5CLambda%20V%5E%7B-1%7D
- ```정방행렬에 대한 고윳값-고유벡터의 성질```
    - 고윳값은 복소수이다.
    - 고윳값은 N 개이다. (N 차원 정방행렬의 고윳값은 N 개이다.)
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20AV%3D%5CLambda%20V
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20trA%3D%5Csum_%7Bi%3D1%7D%5EN%5Clambda_i , (대각합은 고윳값들의 합과 같다.)
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20detA%3D%5Cprod_%7Bi%3D1%7D%5EN%5Clambda_i , (행렬식은 고윳값들의 곱과 같다.)

#### 대각화 가능
- ```행렬이 대각화가능 하려면 고유벡터가 선형독립이어야 한다.```
    - 고유벡터행렬 V 가 역행렬이 있어야 하므로, 역행렬의 조건인 풀랭크이어야 한다. 정방행렬의 풀랭크 조건은 벡터들이 선형독립이어야 한다.

#### 고윳값과 역행렬
- ```대각화가능한 행렬에 0인 고윳값이 없으면 항상 역행렬이 존재한다.```
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%5E%7B-1%7D%3D%28V%5CLambda%20V%5E%7B-1%7D%29%5E%7B-1%7D%3DV%5CLambda%5E%7B-1%7D%20V%5E%7B-1%7D
    - 고윳값행렬은 고윳값의 대각행렬이다. 대각행렬의 역행렬은 대각요소들의 역수이므로, 고윳값이 0이 있으면 역행렬을 만들 수 없다.
    - 또한 A 의 행렬식은 고윳값의 곱과 같다는 정의에 의해서 고윳값 중에 0 이 있으면, 행렬식 값이 0 이므로 역행렬이 존재하지 않는다.

#### 대칭행렬의 고유분해
- 대칭행렬의 성질 : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20S%5ET%3DS%2C%5C%2C%5C%2C%20S%5Cin%5Cmathbf%20%7BR%7D%5E%7BN%5Ctimes%20N%7D , (정방행렬만 대칭행렬이 가능하다.)
- ```행렬 A가 실수인 대칭행렬이면, 고윳값이 실수이고, 고유벡터는 서로 직교한다.```
    - 고유벡터가 단위벡터로 정규화 된 상태이면, 고유벡터행렬 V 는 정규직교 행렬이다.
    - 정규직교 행렬의 성질 : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%5ETV%3DVV%5ET%3DI , https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%5E%7B-1%7D%3DV%5ET , (https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_i%5ETv_j%3D0%20%5C%3B%5C%3B%20%28i%5Cneq%20j%29, https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20v_i%5ETv_j%3D1%20%5C%3B%5C%3B%20%28i%3Dj%29)
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%3DV%5CLambda%20V%5ET
    - ```따라서 실수인 대칭행렬은 항상 대각화가능하다.```

#### 대칭행렬과 랭크-1 행렬
- N 차원 대칭행렬 A 를 N 개의 랭크-1 행렬의 합으로 나타낼 수 있다.
- 로우랭크에 속하는 랭크-1행렬은 N 차원 벡터 1개로 만들 수 있는 정방행렬이다. rank=1
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A_i%3Dv_iv_i%5ET
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%3D%5Csum_%7Bi%3D1%7D%5EN%5Clambda_i%20v_iv_i%5ET%3D%5Csum_%7Bi%3D1%7D%5EN%5Clambda_iA%3D%5Clambda_1%20A_1&plus;%5Ccdots&plus;%20%5Clambda_N%20A_N
- 만약 고윳값 중에 0 이 없다면, 역행렬도 랭크-1 행렬로 나타낼 수 있다.
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%5E%7B-1%7D%3DV%5CLambda%5E%7B-1%7D%20V%5ET%3D%5Csum_%7Bi%3D1%7D%5EN%20%5Cdfrac%7B1%7D%7B%5Clambda_i%7Dv_iv_i%5ET%3D%5Cdfrac%7B1%7D%7B%5Clambda_1%7DA_1&plus;%5Ccdots&plus;%5Cdfrac%7B1%7D%7B%5Clambda_N%7DA_N

#### 대칭행렬의 고윳값 부호
- 대칭행렬을 랭크-1 행렬의 합으로 나타낼 수 있고, 고유벡터가 서로 직교한다는 성질을 사용하면 다음 정리가 가능하다.
- ```각 정리에 대한 증명 중요!!!```
    - 대칭행렬이 양의 정부호이면, 모든 고윳값이 양의 정부호이다. 역도 성립한다. : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20PS%20%5Cleftrightarrow%20%5Clambda_i%3E0
    - 대칭행렬렬이 양의 준정부호이면, 모든 고윳값은 0 이거나 양수이다. 역도 성립한다. : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20PSD%20%5Cleftrightarrow%20%5Clambda_i%20%5Cge%200
    - 양의 준정부호 정리에서 실제로 고윳값은 0 이 될 수 없다. 벡터 x 와 고유벡터 vi 가 직교할 때 0 이되는데, 고유벡터 v 집합은 N 차원 벡터공간에서 기저벡터를 이루기때문에 모든 기저벡터와 직교하는 벡터는 존재하지 않는다. 따라서 양의 정부호이다.
- ```대칭행렬에 대한 고윳값-고유벡터의 성질```
    - 고윳값은 실수이다.
    - 고유벡터행렬의 전치연산은 역행렬과 같다. : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%5ET%20%3D%20V%5E%7B-1%7D
    - 행렬 A 는 고유벡터행렬과 고윳값행렬, 고유벡터행렬의 전치연산의 곱이다. : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%3DV%5CLambda%20V%5ET
    - 행렬 A 는 랭크-1 행렬의 합으로 나타낼 수 있다. : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20A%3D%5Csum_%7Bi%3D1%7D%5EN%5Clambda_i%20v_iv_i%5ET

#### 분산행렬
- ```분산행렬 scatter matrix``` : 임의의 실수 행렬 X 에 대하여 https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20X%5ETX 인 정방행렬. 확률분포에서 사용된다.
- 분산행렬의 정리
    - ```분산행렬은 양의 준정부호이고 고윳값은 0보다 크거나 같다.```
    - 분산행렬의 이차형식으로 증명 : https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5ET%28X%5ETX%29x%3D%28Xx%29%5ET%28Xx%29%3Du%5ETu%3D%5C%7Cu%5C%7C%5E2%20%5Cge%200
    - 분산행렬의 이차형식을 정리하면 어떤 행렬 u 의 제곱합이 되는데, 제곱합은 0 보다 크거나 같으므로 준정부호이다.
    - 또한 분산행렬은 대칭행렬이므로, 대칭행렬이 양의 준정부호이면 고윳값들은 모두 0 보다 크거나 같다.

#### 분산행렬의 역행렬
- ```행렬 X 가 풀랭크이면 이 행렬의 분산행렬의 역행렬이 존재한다.```
    - 행렬 X 가 풀랭크이면 열벡터들이 선형독립이면서 벡터공간의 기저벡터이다.
    - 행렬 x 의 열벡터인 v 벡터가 영벡터가 아니라면, Xv=u 가 성립하며 이때 어떤 벡터 u 는 영벡터가 아니다.
    - u 를 영벡터로 만드는 v 벡터가 영벡터가 아닌 벡터라고 한다면 선형독립이 아니게 된다. 따라서 X 가 풀랭크이므로 v 는 선형독립하고, v가 영벡터가 아니면 u 도 영벡터가 아니다. 그러므로 분산행렬의 이차형식은 양의 정부호가 된다.
    - https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20x%5ET%28X%5ETX%29x%3D%28Xx%29%5ET%28Xx%29%3Du%5ETu%20%3E%200
    - ```분산행렬이 양의 정부호이면 항상 역행렬이 존재한다.```
        - 분산행렬은 대칭행렬이므로 대칭행렬이 양의 정부호이면 고윳값은 모두 0보다 크다.
        - 고윳값이 모두 양수이면 모든 고윳값의 곱이 양수이며 0보다 크다.
        - 정방행렬의 성질에 의해서 행렬식의 값이 모든 고윳값의 곱과 같으므로, 행렬식은 0 보다 크다. 따라서 역행렬이 존재한다.
- 역행렬이 존재하면 대칭행렬은 항상 양의 정부호인가?
    - 그렇지 않다. 역행렬이 존재한다면 행렬식이 0 이 아닌 양수이거나 음수이다. 행렬식이 음수인 경우라면, 고윳값의 곱이 음수이므로, 고윳값 중 한 개 이상이 음수가 된다.
    - 고윳값과 대칭행렬의 부호 성질에 의해서 고윳값이 음수이면 대칭행렬은 양의 정부호가 아니다.
    - 그러므로 역행렬이 존재한다고 해서 대칭행렬이 항상 양의 정부호 인것은 아니다.
    - 대칭행렬이 양의 정부호이면 항상 역행렬이 존재한다는 맞다.

#### 고유분해의 성질 요약
- 고유분해와 관련된 정리들은 데이터분석에서 자주 사용되므로 잘 익혀야 한다.
- N 차원 정방행렬 A 에 대해서 다음과 같은 사항이 성립한다.
    - 행렬 A는 N개의 고윳값-고유벡터를 갖는다. (복소수인 경우와 중복고윳값인 경우 포함)
    - 행렬의 대각합은 모든 고윳값의 합과 같다.
    - 행렬의 행렬식은 모든 고윳값의 곱과 같다.
    - 행렬 A가 대칭행렬이면 실수 고윳값 N 개를 가지며 고유벡터들이 서로 직교한다.
        - 정규직교하는 벡터의 성질은 증명에서 잘 사용된다. https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%5ETV%3DI%2C%5C%3B%5C%3BV%5ET%3DV%5E%7B-1%7D%2C%20%5C%3B%5C%3Bv_i%5ETv_j%3D0%28i%20%5Cneq%20j%29%2C%5C%3B%5C%3Bv_i%5ETv_j%3D1%28i%3Dj%29
    - 행렬 A가 대칭행렬이고 고윳값이 모두 양수이면 양의 정부호이고 역행렬이 존재한다. 역도 성립한다.
    - 행렬 A가 어떤 행렬 X의 분산행렬이면 0 또는 양의 고윳값을 가진다.
    - 행렬 X가 풀랭크이면 분산행렬의 역행렬이 존재한다.

