# partial()
- functools 라이브러리의 모듈 중 하나
- 원래 함수를 바꾸지 않고 함수의 파라미터 값을 변경해준다.

### 사용 방법
- 모듈 임포트
```
from functools import partial
```

#### 함수 정의
- 문자열을 인덱스 5까지만 반환해주는 함수
```
def text_cutter(text, cut=5) :
    new_text = text[:cut]
    return new_text
```
- 함수 테스트
```
text_list = "1234567890"
text_cutter(text_list)

=====<print>=====

'12345'
```

#### 함수의 파라미터 값을 변경
- 함수의 파라미터 cut의 값은 5로 정의되어 있다. partial의 인수로 cut의 값을 재 정의 할 수 있다.
- cut을 3으로 재정의한 뒤 함수를 호출하면 출력되는 문자의 길이가 바뀌어 있는 것을 볼 수 있다.
```
text_list = "1234567890"

text_cutter_v1 = partial(text_cutter, cut=3)

text_cutter_v1(text_list)

=====<print>=====

'123'
```

#### 함수의 파라미터를 여러가지로 변경
- partial() 을 사용하여 파라미터 값을 순차적으로 바꾸고, 반환되는 값을 확인
- partial()의 속성
   - new_text_cutter.keywords : 변경한 파라미터의 이름과 값을 반환해준다.
   - new_text_cutter.func : 새로 정의된 함수의 내부주소 갑과 원래 함수의 이름을 반환해준다.
   - new_text_cutter.args : 새로 정의 된 함수의 이름을 반환해준다.

```
text_list = "1234567890"

for i in range(1, 11) :
    new_text_cutter = partial(text_cutter, cut=i)
    print(new_text_cutter(text_list), " // ", new_text_cutter.keywords, " // ", new_text_cutter.func, " // ", new_text_cutter.args)

=====<print>=====

1  //  {'cut': 1}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
12  //  {'cut': 2}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
123  //  {'cut': 3}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
1234  //  {'cut': 4}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
12345  //  {'cut': 5}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
123456  //  {'cut': 6}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
1234567  //  {'cut': 7}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
12345678  //  {'cut': 8}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
123456789  //  {'cut': 9}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
1234567890  //  {'cut': 10}  //  <function text_cutter at 0x000002A1C6BF59D8>  //  ()
```
