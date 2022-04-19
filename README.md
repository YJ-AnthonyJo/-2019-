# EducationDonationFair_2019
## 공통
드론 : Tello를 사용하였다. 이 드론은 네트워크 통신을 통해 명령 전달이 가능하다.  
## drone_1pic_follow
하나의 얼굴 사진을 입력하면 얼굴 특징을 추출하여 사용자를 인식하고 따라가도록 한다.  
pretrained model인 vgg_face_weights.h5를 사용하였으나 깃허브상 100MB이상의 파일을 올릴 수 없으므로, 링크로 대체한다.<a href link="https://www.kaggle.com/datasets/acharyarupak391/vggfaceweights">링크</a>
## drone_need_train
동작(몸짓)인식으로 드론을 조종하는 프로젝트이다.  
데이터 : 당시 파란색 티셔츠를 이용하여 opencv를 통해 티셔츠 부분만을 추출하여 모션인식에 활용하였다.
