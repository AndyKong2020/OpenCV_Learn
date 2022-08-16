# Opencv学习笔记
## 第一个程序：打开一张图片
```python
import cv2

img = cv2.imread("E:\\3.jpg")
#使用**cv.imread**()函数读取图像。图像应该在工作目录或图像的完整路径应给出。
cv2.namedWindow("image")
cv2.imshow("image", img)
#使用函数**cv.imshow()**在窗口中显示图像。窗口自动适合图像尺寸。
cv2.waitKey(0)
```
加载彩色灰度图像
```python
img = cv2.imread('messi5.jpg',0)
```
## 尝试运行一段代码：获取摄像头图像
```python
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 显示结果帧e
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()
```
要捕获视频，你需要创建一个 VideoCapture 对象。它的参数可以是设备索引或视频文件的名
称。设备索引就是指定哪个摄像头的数字。正常情况下，一个摄像头会被连接(就像我的情况一
样)。所以我简单地传0(或-1)。你可以通过传递1来选择第二个相机，以此类推。在此之后，你可以
逐帧捕获。但是在最后，不要忘记释放俘虏。

## 运行手势识别
```python
# 作者:明月清风我
# 时间:2022/2/25 15:49
import cv2
import mediapipe as mp
import time

# 获取视频对象，0为摄像头，也可以写入视频路径
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# Hands是一个类，有四个初始化参数，static_image_mode,max_num_hands,min_detection_confidence,min_tracking_confidence
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # 画线函数

pTime = 0  # 开始时间初始化
cTime = 0  # 目前时间初始化

while True:
    # sucess是布尔型，读取帧正确返回True;img是每一帧的图像（BGR存储格式）
    success, img = cap.read()
    # 将一幅图像从一个色彩空间转换为另一个,返回转换后的色彩空间图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 处理RGB图像并返回手的标志点和检测到的每个手对象
    results = hands.process(imgRGB)
    # results.multi_hand_landmarks返回None或手的标志点坐标
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # landmark有21个（具体查阅上面的参考网址），id是索引，lm是x,y坐标
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # lm的坐标是点在图像中的比例坐标
                # h-height,w-weight图像的宽度和高度
                h, w, c = img.shape
                # 将landmark的比例坐标转换为在图像像元上的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # 将手的标志点个性化显示
                cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
            # 在图像上绘制手的标志点和他们的连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 将帧率显示在图像上
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 1)
    # 在Image窗口上显示新绘制的图像img
    cv2.imshow("Image", img)
    # 这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户按下键，则继续执行后面的代码，如果用户没有按下键,则接续等待
    if cv2.waitKey(1) == ord('q'):
        break
```

## 捕获面部
```python
# 作者:明月清风我
# 时间:2022/2/26 9:17
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
```

## 运用pyautogui调节音量
```python
import pyautogui
#pyautogui.press('volumeup')    #调大音量
pyautogui.press('volumedown') #调小音量
#pyautogui.press('volumemute') #静音
```

## 手势操作鼠标
```python
# 作者:明月清风我
# 时间:2022/2/25 15:49
import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
import math

# 获取视频对象，0为摄像头，也可以写入视频路径
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# Hands是一个类，有四个初始化参数，static_image_mode,max_num_hands,min_detection_confidence,min_tracking_confidence
hands = mpHands.Hands(False, 1, 0.2, 0.4)
mpDraw = mp.solutions.drawing_utils  # 画线函数

pTime = 0  # 开始时间初始化
cTime = 0  # 目前时间初始化

while True:
    # sucess是布尔型，读取帧正确返回True;img是每一帧的图像（BGR存储格式）
    success, img = cap.read()
    #print(img.shape)
    img = cv2.flip(img, 1)
    # 将一幅图像从一个色彩空间转换为另一个,返回转换后的色彩空间图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理RGB图像并返回手的标志点和检测到的每个手对象
    results = hands.process(imgRGB)
    # results.multi_hand_landmarks返回None或手的标志点坐标
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # landmark有21个（具体查阅上面的参考网址），id是索引，lm是x,y坐标
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # lm的坐标是点在图像中的比例坐标
                # h-height,w-weight图像的宽度和高度
                h, w, c = img.shape
                # 将landmark的比例坐标转换为在图像像元上的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:
                    dx, dy = cx, cy

                if id == 8:
                    pyautogui.moveTo(x = cx*3, y = cy*2.25)

                    print(cx*3, cy*2.25)

                    p1 = np.array([dx, dy])
                    p2 = np.array([cx, cy])
                    p3 = p2 - p1
                    p4 = math.hypot(p3[0], p3[1])

                    if p4 < 20:
                        pyautogui.click()
                        print("click")


                # 将手的标志点个性化显示
                cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
            # 在图像上绘制手的标志点和他们的连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 将帧率显示在图像上
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 1)
    # 在Image窗口上显示新绘制的图像img
    cv2.imshow("Image", img)
    # 这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户按下键，则继续执行后面的代码，如果用户没有按下键,则接续等待
    if cv2.waitKey(1) == ord('q'):
        break
```
食指和拇指捏合点击，食指尖是鼠标坐标

## 使用拇指和中指的捏合来控制音量
```python
# 作者:明月清风我
# 时间:2022/2/25 15:49
import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
import math

# 获取视频对象，0为摄像头，也可以写入视频路径
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# Hands是一个类，有四个初始化参数，static_image_mode,max_num_hands,min_detection_confidence,min_tracking_confidence
hands = mpHands.Hands(False, 1, 0.1, 0.1)
mpDraw = mp.solutions.drawing_utils  # 画线函数

pTime = 0  # 开始时间初始化
cTime = 0  # 目前时间初始化

while True:
    # sucess是布尔型，读取帧正确返回True;img是每一帧的图像（BGR存储格式）
    success, img = cap.read()
    #print(img.shape)
    img = cv2.flip(img, 1)
    # 将一幅图像从一个色彩空间转换为另一个,返回转换后的色彩空间图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理RGB图像并返回手的标志点和检测到的每个手对象
    results = hands.process(imgRGB)
    # results.multi_hand_landmarks返回None或手的标志点坐标
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # landmark有21个（具体查阅上面的参考网址），id是索引，lm是x,y坐标
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # lm的坐标是点在图像中的比例坐标
                # h-height,w-weight图像的宽度和高度
                h, w, c = img.shape
                # 将landmark的比例坐标转换为在图像像元上的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:
                    dx, dy = cx, cy

                if id == 8:
                    #pyautogui.moveTo(x = cx*3, y = cy*2.25)

                    #print(cx*3, cy*2.25)

                    p1 = np.array([dx, dy])
                    p2 = np.array([cx, cy])
                    p3 = p2 - p1
                    p4 = math.hypot(p3[0], p3[1])

                    if p4 < 20:
                        #pyautogui.click()
                        print("click")

                    ex, ey = cx, cy

                if id == 12:
                    p1 = np.array([dx, dy])
                    p2 = np.array([cx, cy])
                    p3 = p2 - p1
                    p4 = math.hypot(p3[0], p3[1])

                    if p4 < 20:
                        print(cy)
                        if cy < 120:
                            pyautogui.press('volumeup')
                        else:
                            pyautogui.press('volumedown')


                # 将手的标志点个性化显示
                cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
            # 在图像上绘制手的标志点和他们的连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 将帧率显示在图像上
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 1)
    # 在Image窗口上显示新绘制的图像img
    cv2.imshow("Image", img)
    # 这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户按下键，则继续执行后面的代码，如果用户没有按下键,则接续等待
    if cv2.waitKey(1) == ord('q'):
        break
```

## 这只是一个鼠标连点器
```python
# 作者:明月清风我
# 时间:2022/3/3 22:39
import pyautogui as pg
while True:
    pg.click()
```

## 键盘的自动输入
对于不可粘贴的网站，使用自动输入
```python
# 作者:明月清风我
# 时间:2022/3/6 21:25

"""题目：
Directions: For this part, you are allowed 30 minutes to write an essay related to the short passage given below. In your essay, you are to comment on the phenomenon described in the passage and suggest measures to address the issue. You should write at least 150 words but no more than 200 words.

Nowadays star chasing is prevalent among many teenagers.
They take pop stars as their idols, imitating their way of talking,
following their style of dressing,
and seeking every chance to meet them in person at great expenses."""

import pyautogui as pg
import time

essay = ""

time.sleep(5)
pg.typewrite(essay, 0.01)


```