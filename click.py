# 作者:明月清风我
# 时间:2022/3/3 22:39
import cv2 as cv
import pyautogui as pg
while True:
    if cv.waitKey(1) == ord('p'):
        while True:
            pg.click()
            if cv.waitKey(1) == ord('p'):
                break