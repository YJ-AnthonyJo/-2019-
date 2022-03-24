# EducationDonationFair_2019
## 공통
모델 부분 : 기존에 사용하던 코드 일부수정  
## drone_1pic_follow
하나의 얼굴 사진을 입력하면 얼굴 특징을 추출하여 사용자를 인식하고 따라가도록 한다.  
pretrained_model인 vgg_face_weights.h5를 사용하였으나 깃허브상 100MB이상의 파일을 올릴 수 없으므로, 링크로 대체한다.<a href link="https://www.kaggle.com/datasets/acharyarupak391/vggfaceweights">링크</a>
## drone_need_train
동작(몸짓)인식으로 드론을 조종하는 프로젝트이다.  
데이터 : 당시 학교 체육복이 모두 파란색인점을 이용하여 opencv를 통해 체육복 부분만을 추출하여 모션인식에 활용하였다.
