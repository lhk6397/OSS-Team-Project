# OSS PROJECT PROPOSAL
## Team Name : REAL KK
## Project Name : 야구마구


## 팀원 
- 컴퓨터공학과 20101214 김윤형 [아이디어 기획]
- 컴퓨터공학과 20100697 고지완 [전체 관리]
- 컴퓨터공학과 20101272 이용주 [인터넷을 통한 데이터 수합]
- 컴퓨터공학과 20101206 김문정 [결과물 확인 및 피드백]

## Project Goal
롯데자이언츠의 21시즌 실제 경기기록을 머신러닝을 이용한 승부예측 결과와 비교 

## Why
롯데자이언츠 팬으로서 패배의 요인을 심도있게 분석하고, 반성의 시간을 가짐.
또한, 수업시간에 배운 머신러닝을 실제로 구현하여 그 예측값의 정확도가 높은지 측정하고자 함.
실제 경기기록의 지표값들(AVG, OPS, ERA 등)이 실제 경기 결과에 얼마나 영향을 미치는지 확인.

## ToDo List
- STATIZ 사이트를 통해 데이터 수합(타격 AVG, 타격 OPS, 투구 ERA, 투수 WHIP, 수비 A, 수비 E, 홈/어웨이, 상대승률(작년기준))
- 프로그램 구현 (SVM 이용)
- 검증 후 피드백
- 버그 수정, 보완 및 게시

## FeedBack
- 초기에 인공지능이 산출한 결과를 살펴보았을 때 그 값중 동점을 나타내는 2라는 값이 존재하지 않았고 승패에 대한 값인 0과 1로만 결과가 나왔다. 즉, 인공지능이 동점에 대한 연산을 하지 못했다. 문제의 원인을 정확히 파악하기 위해 동점인 경우 또한 승리의 경우인 1로 판단하도록 처리했더니 정확도가 47%에서 70%까지 올라가는 것을 확인했다. 이를 통해 동점처리가 문제의 원인임을 파악했다. 이를 해결할 수 있는 방안으로 득실점을 속성으로 추가하고자 했으나 이긴 경기의 득실점은 양수이고 진 경기의 득실점은 음수로 설정하면 경기의 승패여부를 인공지능에게 공개하는 것이 되므로 득실점의 절대값을 속성으로 추가하기로 했다. 이를 추가하여 다시 연산을 수행한다면 동점일 때 득실점의 절대값이 0임을 이용해서 동점의 처리를 수행하고 정확도가 올라갈 것으로 예상했다. 하지만 예상과 달리 정확도가 46%로 낮아졌고 동점에 대한 처리 또한 해결하지 못했다.
- 인공지능이 동점에 대해 처리하지 못한 이유를 생각해본다면 먼저 인공지능이 학습하도록 입력한 속성들(타율 AVG, 투수 ERA 등등)이 동점을 판단하기에는 적절하지 않은 지표값인 것으로 예상된다. 위의 방법을 통해 이를 해결하고자 했으나 한 시즌에 진행한 약 140개의 경기만으로는 인공지능이 학습하기에 부족한 데이터 갯수였을 것이다.
- csv 
- 자연현상
