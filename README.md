# Reconstructing image from augmented segments
## [2x2]
MNIST 데이터를 활용하여 해당 과제를 시작했습니다.

## [3x3]

처음에는 각 segment의 edge들 간의 distance가 가장 작은 augmentation을 답안으로 제시하였으나 픽셀 특성상 MNIST의 흰색 숫자 부분이 아닌 검은 바탕 부분 (pixel=0)을 이어붙인 이미지가 정답으로 제시되어 다른 방법 물색했습니다.







배운점: argparse는 bool 타입으로 못받고 int 아니면 str
      2by2까지는 모든 augmentation 경우의 수를 완전 탐색하여 진짜 이미지일 확률이 가장 높은 이미지를 답으로 정했는데 3by3부터는 모든 augmentation 경우의 수가 너무 많아서 분류 딥러닝 모델을 사용하기보다 이미지 생성형 모델을 학습시켜 사용하는 것이 낫다는 점

한계점: MNIST 데이터로만 진행했다는 점