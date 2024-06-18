## 주식시장 예측
NIFTY 50 지수에 포함된 50개 종목의 가격 내역과 거래량 모든 데이터 세트는 일 단위, 각 주식에 대한 가격 및 거래 값이 각 주식에 대한 csv파일과 주식 자체에 대한 일부 매크로 정보 포함

### 시계열 분석
# 주식 가격 시계열 분석에는 어떤 데이터가 필요할까?
# 일단 데이터를 하나 까볼까?
df_adaniports['Close'].hist()
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/54902356-4df3-486e-9316-fcbb5fbf3715)

# 주식 예측 한 걸 확대해보면
# 28일의 주식 가격 어떻게 될 거 같냐 할 때 27일이랑 같다. 라고 답이 변한다
# 왜 이런 현상이 나올까
# loss값 최소화 -> (예측값 - 실제값)
# 이런 최소화할 때 내일 주식가격은 오늘주식가격과 똑같을 것이다
# 가 가장 로스가 낮음 - 쓸데없죠


### 해결책
1. 주식 가격을 sequence 데이터로 lstm에 집어넣는 게 맞는지 고민
1~10일 주식 가격이랑 11일 주식 가격이 연관성이 있나?
다른 게 더 높을 수 있다 영향이
주식가격은 랜덤성이 짙어서 예측할 수 없다
모든 주식관련 정보가 주식가격에 선반영이 되어 있다

2. 가격이 아니라 다른 결과 보는 모델이라면?
수익률로 본다면, 20일간 추이를 집어놓고 다음날의 가격이 아니라 다음날 주식을
사는 게 맞는지 파는 게 맞는지 예측 -> sequential의 컨닝을 막을 수 있지 않을까

3. 가격 말고 다른 요인을 넣는 거는?
전날 거래량, SNS 언급량, 전날 나스닥지수 증감량, 관련 업종 증가추이 등 여러 요소가있다.
복잡한 레이어를 구성하고 싶으면 Functional API(what?)를 쓰거나, 
feature column(이게 뭘까?)을 이용하면 인풋이 몇 개든 몇 종류든 쉽게 집어넣을 수 있다
하지만 현재는 캐글 대회에서 따온 데이터셋을 이용해 시계열 데이터 분석을 진행한 것으로, 굳이이다.

4. stationary 된 데이터를 사용하자
주식가격은 시계열 데이터이다.
예측모델을 만들고 싶을 땐 이 데이터가 stationary(평균, 분산, 공분산이 비교적 일정한 데이터) 데이터인지 확인해야 함
이게 아니면 너무 무작위라 예측모델을 애초에 만들 수가 없다 판단
예를 들어 일별 주식가격 변화량으로 바꾸는 것
이걸 따지면 주식가격 말고 다른 가격 변화량도 보면 좋다
stationary인게 검증하려면 Dickey-Fuller 검증이란 것도 있음
(2번, LSTM 매도 매매 모델에서 진행함)


5. sequential 컨닝을 막을 거면, transformer 모델 쓰면 되지 않나??
도전할 필요도 있다..!!

-------------
# 우선, 1번의 문제점인 LSTM을 쓰되, 2번. 즉, 가격이 아닌 매매 또는 매도 를 종속변수로 사용

![image](https://github.com/dbgks25/algorithm-study/assets/58736053/38bb779f-77e0-4bae-a8ac-ae9b7688c84c)
"""
DataFrameSelector : 지정된 열을 선택하는 변환기 - Close 열만 선택
FunctionTransformer : 람다 함수를 사용하여 데이터를 2D 형태로 변환
StandardScaler : 데이터 표준화(각 피처의 평균을 0, 표준편차를 1로 변환)
FunctionTransformer : 데이터를 3D 형태로 변환
CreateSequences : 시퀀스를 생성하하는 변환기. 지정된 시퀀스 길이에 따라 데이터를 시퀀스로 변환
LSTMModel : LSTM 모델을 훈련하고 예측하는 변환기

"""
--> 대략적으로 만든 LSTM 모델
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/3eb4adfb-3693-4d1e-aadf-550848f35b1b)

"""
1. input layer : 시퀀수 길이 20, 피쳐 수 1의 형태
2. LSTM : 첫 번째 LSTM 레이어
시퀀스 길이 20, 피처 수 1의 데이터를 받아 50개의 유닛으로 구성된 LSTM 적용
return_sequences=True 옵션으로 모든 타임스텝에 대한 출력 반환
3. Dropout
드롭아웃 레이어 : 50%의 뉴런을 무작위로 드롭하여 과적합 방지
4. LSTM : 두 번째 LSTM 레이어
return_sequences=False 옵션으로 마지막 타임스텝에 대한 출력만 반환(for predict)
5. Dropout : 두 번째 드롭아웃 레이어로, 50%의 뉴런을 무작위로 드롭
6. Dense : 출력층, sigmoid 활성화 함수를 사용하여 이진 분류 문제 해결

"""
### 결과
Test Loss: 0.6931039094924927, Test Accuracy: 0.5041493773460388
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/81d6d1d7-96b9-424a-9113-941599114d7b)
ㅠㅠ
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/5c4fa0c8-6cee-48be-bd35-2c9165afdd99)
predict에서 생각보다 많은 매도 예측을 함
-----------------
# 이제, LSTM을 사용했으니, Transformer에 대해서 진행

