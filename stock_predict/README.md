## 주식시장 예측
NIFTY 50 지수에 포함된 50개 종목의 가격 내역과 거래량 데이터 세트는 일 단위, 각 주식에 대한 가격 및 거래 값을 포함하며 각 주식에 대한 csv 파일과 주식 자체에 대한 일부 매크로 정보가 포함되어 있습니다.

### 시계열 분석
# 주식 가격 시계열 분석에는 어떤 데이터가 필요할까?
데이터를 하나 까볼까요? Adani Ports의 'Close' 데이터를 사용한 히스토그램입니다.
df_adaniports['Close'].hist()
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/54902356-4df3-486e-9316-fcbb5fbf3715)

### 주식 예측 한 걸 확대해보면
28일의 주식 가격을 예측할 때 27일과 같은 값을 예측하는 경향이 있습니다. 왜 이런 현상이 나올까요? 이는 loss 값을 최소화하려는 경향 때문입니다.
Loss 값 최소화: 예측값과 실제값의 차이를 최소화하려는 것.
내일 주식가격은 오늘 주식가격과 같을 것이다 라는 예측은 가장 낮은 loss를 가질 수 있습니다.

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
하지만 현재는 캐글 대회에서 따온 데이터셋을 이용해 시계열 데이터 분석을 진행한 것

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
- DataFrameSelector: 지정된 열을 선택하는 변환기 - Close 열만 선택.
- FunctionTransformer: 데이터를 2D 형태로 변환.
- StandardScaler: 데이터 표준화(각 피처의 평균을 0, 표준편차를 1로 변환).
- FunctionTransformer: 데이터를 3D 형태로 변환.
- CreateSequences: 시퀀스를 생성하는 변환기. 지정된 시퀀스 길이에 따라 데이터를 시퀀스로 변환.
- LSTMModel: LSTM 모델을 훈련하고 예측하는 변환기.

--> 대략적으로 만든 LSTM 모델
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/3eb4adfb-3693-4d1e-aadf-550848f35b1b)

Input Layer: 시퀀스 길이 20, 피처 수 1의 형태.
LSTM: 첫 번째 LSTM 레이어.
시퀀스 길이 20, 피처 수 1의 데이터를 받아 50개의 유닛으로 구성된 LSTM 적용.
return_sequences=True 옵션으로 모든 타임스텝에 대한 출력 반환.
Dropout: 드롭아웃 레이어로 50%의 뉴런을 무작위로 드롭하여 과적합 방지.
LSTM: 두 번째 LSTM 레이어.
return_sequences=False 옵션으로 마지막 타임스텝에 대한 출력만 반환(for predict).
Dropout: 두 번째 드롭아웃 레이어로, 50%의 뉴런을 무작위로 드롭.
Dense: 출력층, sigmoid 활성화 함수를 사용하여 이진 분류 문제 해결.


### 결과
Test Loss: 0.6931039094924927, Test Accuracy: 0.5041493773460388
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/81d6d1d7-96b9-424a-9113-941599114d7b)
ㅠㅠ
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/5c4fa0c8-6cee-48be-bd35-2c9165afdd99)
predict에서 생각보다 많은 매도 예측을 함
-----------------
### 이제, LSTM을 사용했으니, Transformer에 대해서 진행
트랜스포머 모델은 자연어 처리(NLP) 분야에서 매우 중요한 변화를 가져온 딥러닝 모델이다.

트랜스포머 모델의 주요 개념과 구조

#### 주요 개념
1. 어텐션 메커니즘(Attention Mechanism)
- 입력 문장의 각 단어가 출력 문장의 각 단어에 얼마나 중요한지 가중치 할당(by 내적)
- 셀프 어텐션(Self-Attention)은 문장의 다른 모든 단어와의 관계를 고려하여 각 단어의 표현을 만듦
 (셀프가 아닌 일반 어텐션은 순차적으로 다른 단어와의 관계를 가중치에 계속 합함. --> 마지막으로 가면 갈수록 예측이 어려워진다.)
 
2. 인코더-디코더 구조
- 인코더 : 입력 문장을 처리하여 고정 길이의 표현 벡터 생성
- 디코더 : 표현 벡터를 사용하여 출력 문장 생성


#### 트랜스포머 모델의 구조
1. 인코더:
- 여러 개의 인코더 레이어로 구성되어 있으며, 각 레이어는 셀프 어텐션 및 피드포워드 신경망으로 구성
- 입력 임베딩에 위치 인코딩(Positional Encoding)을 추가하여 단어의 위치 정보 반영

2. 디코더:
- 인코더와 유사하게 여러 디코더 레이어로 구성되어 있으며, 셀프 어텐션, 인코더-디코더 어텐션 및 피드포워드 신경망으로 구성됨
- 디코더 레이어는 이전 단어의 출력과 인코더의 출력을 함께 사용하여 다음 단어 예측

트랜스포머 모델은 BERT, GPT, T5 등 다양한 파생 모델의 기초
(BERT에 관심이 좀 갑니다.)

그럼,
 시계열 분석에서 왜 유리한가?
트랜스포머 모델은 여러 개의 어텐션 헤드를 사용하여 입력 데이터의 다양한 패턴을 동시에 학습할 수 있음
Informer 모델: 시계열 예측에 특화된 트랜스포머 변형 모델로, 롱 시퀀스의 시계열 데이터를 효과적으로 처리하며 더 효율적인 메모리 사용과 계산 속도를 제공
Temporal Fusion Transformer (TFT): 이 모델은 시계열 데이터를 위한 다중 해석 가능성 및 예측 정확성을 동시에 제공하는 모델로, 여러 입력 특성과 시계열 데이터를 융합하여 예측

<본 프로젝트에서 주가 예측할 때 사용한 모델 구조>
1. Embedding
self.embedding = nn.Linear(1, embed_dim)

2. 다중 헤드 어텐션(Multihead Attention)
self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)

3. 피드포워드 네트워크(Feed-Forward Network)
self.ffn = nn.Sequential(
    nn.Linear(embed_dim, ff_dim),
    nn.ReLU(),
    nn.Linear(ff_dim, embed_dim)
)

4. 레이어 정규화 및 드롭아웃
self.layernorm1 = nn.LayerNorm(embed_dim)
self.layernorm2 = nn.LayerNorm(embed_dim)
self.dropout = nn.Dropout(0.1)

5. 변환
self.flatten = nn.Flatten()
self.dense = nn.Linear(embed_dim * seq_length, 1)
![image](https://github.com/dbgks25/algorithm-study/assets/58736053/8a88df5b-25b9-4957-8551-f2e8f2f5bf8d)



아뿔싸, 인코더 디코더 설명 실컷 해놓고, 모델에선 인코더만 사용했다....
하지만, 인코더 온리 모델, 또는 디코더 온리 모델 또한 있다고 한다!

[Encoder Only 와 Decoder Only 언어모델에 대한 고찰]
https://medium.com/@hugmanskj/encoder-only-%EC%99%80-decoder-only-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B3%A0%EC%B0%B0-9852213dbb72
인코더(Encoder): 심볼(Symbol) → 벡터(Vector) -> 숫자로 가공해 회귀 분류 등 모델
디코더(Decoder): 심볼(Symbol) → 심볼(Symbol)

인코더 할 수 있는 일
분류: 텍스트를 특정 카테고리로 분류하는 작업
회귀분석: 텍스트에서 연속적인 값을 예측하는 작업
임베딩: 텍스트를 고차원 벡터로 변환하는 작업
문서 유사도 측정: 문서 간의 유사도를 평가하는 작업
감정 분석: 텍스트의 감정을 분석하는 작업
주제 모델링: 문서의 주제를 식별하는 작업
추출형 질의 응답: 질문에 적절한 답변을 찾는 작업 (SQUAD 스타일)

인코더 할 수 없는 일
인코더는 N개의 연속된 심볼을 생성하는 작업은 수행할 수 없다. 언어 생성은 불가능하거나 매우 약한 성능을 보이며, 본 프로젝트에서는 단일독립변수이기에 인코더 온리가 성공적으로 결과가 나온 것이다.

디코더 할 수 있는 일
심볼에서 심볼로 변환하는 작업과 관련된 모든 것
번역: 한 언어의 텍스트를 다른 언어로 번역하는 작업
요약: 긴 텍스트를 간결하게 요약하는 작업
작문: 새로운 텍스트를 창작하는 작업
언어 생성: 자연스럽고 일관된 텍스트를 생성하는 작업
채팅봇 개발: 대화형 응용 프로그램을 만드는 작업
문장 완성: 주어진 문장을 완성하는 작업
임베딩: 텍스트를 벡터로 변환하는 작업
 
디코더 할 수 없다고 여겨졌던 일
분류: 텍스트를 특정 카테고리로 분류하는 작업
회귀분석: 텍스트에서 연속적인 값을 예측하는 작업
임베딩: 텍스트를 고차원 벡터로 변환하는 작업
문서 유사도 측정: 문서 간의 유사도를 평가하는 작업
감정 분석: 텍스트의 감정을 분석하는 작업
주제 모델링: 문서의 주제를 식별하는 작업
하지만, 디코더가 인코더도 잘하는 것을 보여주는 모델이 나오고 있다.





