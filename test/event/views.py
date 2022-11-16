from django.shortcuts import render, redirect
from event.models import Post
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/my_model.h5')

# Create your views here.

def index(request):
    if request.method == "GET":
        return render(request,'index.html')
    elif request.method=="POST":
        post = Post()
        post.title = request.POST['title']
        post.image = request.FILES['image']
        post.save()
        post.image 
        return redirect('/result/',{'post':post})

def results(request, post):
    img = cv2.imread(post.image) 
    #이미지 흑백처리 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #이미지 블러 
    img = cv2.GaussianBlur(img, (5,5), 0)
    #이미지 내의 경계 찾기 
    ret, img_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #경계를 직사각형으로 찾기 
    rects = [ cv2.boundingRect(each) for each in contours]
    #직사각형 정렬, 두번째 원소기준으로 정렬
    rects.sort(key=lambda x: (x[1], x[0]))
    #리스트 5개 단위로 나누고, 정렬, 다시 합치기
    def list_chunk(lst, n):
        lists_chunked = [lst[i:i+n] for i in range(0, len(lst), n)]
        imgs_chunked = list(map(sorted, lists_chunked))
        rects = sum(imgs_chunked, [])
        return rects
    rects = list_chunk(rects, 5)
    #추출한 숫자 영역 전처리 
    img_classify = img.copy()

    #최종 이미지 파일용 배열 
    mnist_imgs =[]
    margin_pixel = 15

    #숫자 영역 추출 및 reshape(28,28,1)

    for rect in rects:
        im = img_classify[rect[1]-margin_pixel:rect[1]+rect[3]+margin_pixel, rect[0]-margin_pixel:rect[0]+rect[2]+margin_pixel]
        row, col = im.shape[:2]

    # 정방형 비율을 맞춰주기 위해 변수 이용
    bordersize = max(row, col)
    diff = min(row,col)

    # 이미지의 intensity의 평균을 구함
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    # border 추가해 정방형 비율로 보정 
    border = cv2.copyMakeBorder(
            im,
            top=0,
            bottom=0,
            left = int((bordersize-diff)/2),
            right = int((bordersize-diff)/2),
            borderType = cv2.BORDER_CONSTANT,
            value = [mean, mean, mean]
        )

    square=border
    
    #square 사이즈 (28,28)로 
    resize_img = cv2.resize(square, dsize=(28,28), interpolation = cv2.INTER_AREA)
    mnist_imgs.append(resize_img)
    for i in range(len(mnist_imgs)):
        img = mnist_imgs[i]
        
    #사이즈 변환
    img = img.reshape(-1,28,28,1)
    #데이터 정규화 
    input_data = ((np.array(img)/255) - 1) * -1
    
    # 결과 
    res = np.argmax(model.predict(input_data), axis = -1)
    return render(request,'show.html',{'res':res})
    