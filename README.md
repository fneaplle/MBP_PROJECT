1. layers.py
	한 개의 레이어들을 정의해 놓은 것이다.

2. module.py 
	generator, discriminator, loss 등을 정의해 놓은 것이다.

3. pipeline.py
	데이터의 파이프 라인을 정의 해 놓은 곳이다.

4. test.py
	실제 완성된 모델의 weight를 불러서 테스트 데이터에 대해서 실행해 보는 곳이다.

5. train.py
	학습데이터로 학습대상 모델을 학습하는 공간이다.

## TODO
- [x] 지금은 generator, discriminator를 그저 데모로만 만들어 놓았다.
- [x] generator의 경우 U-Net으로, discriminator의 경우 PathGAN 형태로 만들어서 다시 학습을 진행해야 한다.