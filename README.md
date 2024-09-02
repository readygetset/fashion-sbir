# 패션의 완성은 그림

📢 20##년 1/여름/2/겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 20##년 1/여름/2/겨울학기 AIKU Conference 열심히상 수상!

## 소개
패션 아이템을 검색하려고 하면 멈칫할 때가 한 두 번이 아니다. “그 구두.. 뭐라 검색해야 나오지?”, “요즘 이런 스타일이 유행이던데 이름이 뭐더라” 하는 고민들. 이젠 단순히 텍스트로만, 혹은 이미지로만 검색하지 말고 ⭐️내가 그리는 그림⭐️으로 패션 아이템을 검색해보는 건 어떨까?

## 방법론
SBIR(Sketch Based Image Retrieval) 태스크로, 추가적으로 텍스트를 활용하여 원하는 패션 아이템을 찾는 것이 목표이다. 


**▶️ 데이터셋 구축**

이미지-스케치-텍스트 페어 데이터셋을 구축한다. 스케치의 경우, canny edge map과 다양한 후처리 기법을 활용하여 직접 패션 아이템 이미지와 일대일 대응되는 스케치 이미지를 생성하였다. 하얀색 아이템의 경우, 이미지의 배경 또한 하얀색이기 때문에 아이템의 edge map을 선명하게 추출하지 못하는 문제가 있었다. 이를 해결하기 위해 텍스트 데이터에서 'white, bright, light, pale, ivory, cream, sky, gray'의 단어가 존재하는 경우, 다른 방법의 후처리를 활용하였다.
후처리로는 clahe와 히스토그램 평활화 기법을 활용하였다.

**▶️ 모델링**

CLIP의 vision, text encoder를 사용하여 이미지, 스케치, 텍스트의 임베딩을 구한다. 후에 스케치와 텍스트는 임베딩 값을 혼합한 후에 이미지 임베딩과 Contrastive learning 방식을 통해 metric learning으로 모델을 학습시킨다. 
 
- 이미지와 스케치에 사용하는 vision encoder는 shared weights인 인코더를 사용
- 스케치와 텍스트 임베딩은 avg, concat 등의 방식으로 fuse
- loss로는 infoNCE loss를 활용하여 이미지 임베딩과 스케치+텍스트 임베딩 사이에 대조 학습

## 환경 설정

Prerequisites:
- Pytorch
- ftfy

## 사용 방법

Retrieval_Demo.ipynb 실행

## 예시 결과
**1. edge map과 text caption을 주었을 경우**
![image](https://github.com/user-attachments/assets/2b06e9aa-8e01-4a36-9a80-18f7dbbd57e2)
![image-2](https://github.com/user-attachments/assets/0b12b611-b2d0-4fa6-86d1-4bd49ed4790a)
edge map과 text caption을 함께 주었을 때 원하는 이미지가 가장 먼저 검색되는 것을 확인할 수 있으며, 그 외 사진들도 상당히 연관성이 높은 이미지들임을 알 수 있다. 

**2. 동일한 텍스트에 색깔을 달리 했을 경우**
![image-3](https://github.com/user-attachments/assets/6072a7d9-00fc-41b2-bb8c-b05a6d69ff65)
‘a long trouser’와 바지 스케치를 주었을 경우 첫 번째 row의 왼쪽과 같은 결과를 얻는다. 이때 ‘a long {color} trouser’로 텍스트 캡션에 변화를 주면 아래와 같이 색깔에 잘 맞는 이미지가 검색되는 것을 볼 수 있다. 

**3. 브랜드 이름을 텍스트에 주는 경우**
![image-4](https://github.com/user-attachments/assets/054911cb-6a76-49ef-b38f-0a9e6b35fdd4)
이번에는 데이터셋 특성상 text description에 명품 브랜드명이 있었기 때문에 실험 중 하나로 브랜드명을 넣은 캡션을 제공해보았다. 결과는 브랜드 명에 맞는 이미지가 잘 검색되고 있음을 알 수 있다. 특히나 '루이비통'의 경우 특정 패턴이 돋보이면서 검색 결과가 더 잘 나오는 걸 볼 수 있다.

**4. 캡션 유무에 따른 이미지 검색의 차이**
![image-5](https://github.com/user-attachments/assets/f183cd00-09c1-417c-b340-887485aaccda)
이번에는 캡션의 영향을 알아보기 위해 캡션의 유무에 따른 실험을 진행하였다. 동일한 클러치백이 있는 스케치를 주고 하나는 캡션을 공백으로 주고, 하나는 간단히 'a cluth' 로 주었을 때 결과는 완전히 달랐다. 캡션을 아예 주지 않으면 클러치나 비슷한 물체를 찾지 못했지만, 클러치라고 명시해주면 클러치 가방, 혹은 비슷한 가방 류를 검색하는 것을 볼 수 있다. 

**5. 데모**
![image-6](https://github.com/user-attachments/assets/eb4f3e4e-0c66-46f2-83eb-e8858bc23f2a)
이번에는 edge map이 아닌 실제 인간이 그린 스케치로 얼마나 좋은 성능을 내는지 확인하기 위해 데모를 제작했다. 아래는 하나의 예시로, '보테가 베네타' 가방을 그렸을 때 나오는 이미지 결과이다. 

## 팀원
- [정혜민] (https://github.com/hmin27) : 모델링, 코드 구현
- [서연우] (https://github.com/readygetset) : 모델 학습, 데모 제작
- [이민하] (https://github.com/mlnha) : 데이터셋 구축
