import cv2
import numpy as np


def process_image(image, width, height):
    # 将图片转为灰度图片，提高卷积速度
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    # 将对图片数组进行二值分化处理，方便后面的卷积，也就是说颜色并没有意义，我们只需要轮廓
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    print(image)
    return image[None, :, :].astype(np.float32)  # 拓展为三位数组，方便后面的pytorch的深度学习
