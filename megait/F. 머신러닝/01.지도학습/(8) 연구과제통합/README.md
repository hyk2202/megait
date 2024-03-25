# 머신러닝 연구과제

## #01. 분류분석 연구과제

KNN, NaiveBayes, DecisionTree, SVM, SGD 분류 분석에 대한 통합 연구과제 입니다.

### 문제 1

타이타닉 데이터셋을 지금까지 학습한 모든 알고리즘으로 분석하고 결과를 비교한 후 각 데이터셋에 어떤 알고리즘을 적용하는 것이 더 좋은 성능을 기대할 수 있는지 결론을 내시오.

#### 요구사항

1. 전처리 과정은 이전 수업 내용과 동일하게 진행합니다.(2차전 내용에 따라 진행)
2. 성능이 가장 좋은 알고리즘에 대한 VIF값이 10을 넘어가는 경우 해당 자료를 삭제하고 다시 진행합니다.

### 문제 2

위스콘신 유방암 데이터셋을 지금까지 학습한 모든 알고리즘으로 분석하고 결과를 비교한 후 각 데이터셋에 어떤 알고리즘을 적용하는 것이 더 좋은 성능을 기대할 수 있는지 결론을 내시오.

#### 요구사항

1. 전처리 과정은 이전 수업 내용과 동일하게 진행합니다.
2. VIF값이 10을 넘어가는 경우 해당 자료를 삭제하고 진행합니다.
3. 성능 측정 기준은 정확도와 재현율로 합니다.


### 문제 3

아래의 데이터는 사과의 크기, 무게, 단맛, 바삭함, 과즙, 숙성도, 신맛, 품질과 같은 세부 정보를 통해 사과의 품질을 좋음(`good`)과 나쁨(`bad`)로 구분하고 있다.

이 데이터 셋을 활용하여 사과를 상품과 하품으로 분류하기 위한 모형을 로지스틱과 KNN으로 구현하고 더 좋은 성능을 보이는 분류 모형과 성능 평가 지표를 제시하시오.

> https://data.hossam.kr/mldata/apple_quality.xlsx
>
> https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

#### 요구사항

1. 데이터 전처리는 스케일링만 수행하는 것으로 제한합니다.
2. 성능이 가장 좋은 알고리즘에 대한 VIF값이 10을 초과하는 경우 해당 변수를 제거하고 진행합니다.

#### 변수 설명

| 변수 | 설명 |
|---|---|
| A_id | 각 과일의 고유 식별자 |
| Size | 과일의 크기 |
| Weight | 과일의 무게 |
| Sweetness | 과일의 단맛 정도 |
| Crunchiness | 과일의 아삭함을 나타내는 질감 |
| Juiciness | 과일의 과즙이 풍부한 정도 |
| Ripeness | 과일이 익는 단계 |
| Acidity | 과일의 산도 정도 |
| Quality | 과일의 전반적인 품질 (`good`, `bad`) |


### 문제 4

아래의 데이터셋은 UCI Machine Learning 저장소에서 제공하는 버섯에 대한 데이터 셋이다.

이 데이터셋은 모든 변수에 대해 라벨링을 수행해야 한다.

이 데이터셋을 사용하여 식용 버섯과 독버섯을 구분할 수 있는 분류 모형을 구현하시오.

어떤 알고리즘을 사용하는 것이 가장 적합할지 결론을 제시하시오.

> https://data.hossam.kr/mldata/mushrooms.xlsx
>
> https://www.kaggle.com/datasets/uciml/mushroom-classification

#### 요구사항

1. 데이터 전처리는 스케일링만 수행하는 것으로 제한합니다.
2. 성능이 가장 좋은 알고리즘에 대한 VIF값이 10을 초과하는 경우 해당 변수를 제거하고 진행합니다.

#### 변수 설명

| 변수 | 설명 |
|---|---|
| class | p-독성, e-식용 |
| cap-shape | 버섯의 모자 모양(bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s) |
| cap-surface | 버섯의 모자 표면(fibrous=f, grooves=g, scaly=y, smooth=s) |
| cap-color | 버섯의 모자 색상 |
| bruises | 버섯에 멍이 들었는지 여부(bruises=t, no=f) |
| odor | 버섯의 냄새 |
| gill-attachment | 버섯의 주름 부착 방식(attached=a, descending=d, free=f, notched=n). |
| gill-spacing | 주름 사이의 간격(close=c, crowded=w, distant=d) |
| gill-size | 주름 크기를 나타냅니다 (broad=b, narrow=n). |
| gill-color | 주름 색상 |
| stalk-shape | 줄기 모양(enlarging=e, tapering=t). |
| stalk-root | 줄기의 뿌리 부분 모양 |
| stalk-surface-above-ring | 반지 위의 줄기 표면 |
| stalk-surface-below-ring | 반지 아래의 줄기 표면 |
| stalk-color-above-ring | 반지 위의 줄기 색상 |
| stalk-color-below-ring | 반지 아래의 줄기 색상 |
| veil-type | 베일(버섯의 부속물)의 종류 |
| veil-color | 베일의 색상 |
| ring-number | 반지의 수(none=n, one=o, two=t). |
| ring-type | 반지의 종류 |
| spore-print-color | 포자 자국의 색상 |
| population | 개체군의 분포 형태 |
| habitat | 버섯이 자라는 환경 |



## #02. 회귀분석 연구과제

주어진 문서는 코로나19 전후 골목상권 매출의 변화 요인을 분석하기 위한 연구 자료에 대한 초안과 데이터이다. 

이 내용을 파악하여 연구 과정을 직접 재현하고 의미있는 결론을 제시하라.

> 데이터와 문서 파일은 수업시간중 제공됩니다.