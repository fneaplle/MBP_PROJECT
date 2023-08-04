## Project Overview
MBP 프로젝트는 ECG, PPG 시그널 데이터만을 활용하여 ABP를 예측하는 프로젝트다.
1D-Unet을 기반으로 만들었으며, 디테일한 데이터 생성을 위해 Patch-GAN을 사용하였다.

## Scripts
```layers.py``` 한 개의 레이어들을 정의해 놓은 것이다.

```module.py``` generator, discriminator, loss 등을 정의해 놓은 것이다.

```pipeline.py```데이터의 파이프 라인을 정의 해 놓은 곳이다.

```test.py```실제 완성된 모델의 weight를 불러서 테스트 데이터에 대해서 실행해 보는 곳이다.

```train.py```학습데이터로 학습대상 모델을 학습하는 공간이다.

## TODO
- [x] 지금은 generator, discriminator를 그저 데모로만 만들어 놓았다.
- [x] generator의 경우 U-Net으로, discriminator의 경우 PathGAN 형태로 만들어서 다시 학습을 진행해야 한다.
