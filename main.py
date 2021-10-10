import cv2
import numpy as np


def process_first_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('Raw image', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray img', img_gray)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 1)
    cv2.imshow('Thresholded', img_thresh)
    cv2.waitKey(0)

    img_v = cv2.dilate(img_thresh, np.ones((5, 5), np.uint8), iterations=1)
    img_v = cv2.erode(img_v, np.ones((9, 9), np.uint8), iterations=1)
    cv2.imshow('Result', img_v)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img1_res.jpg', img_v)

def process_second_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('Raw image', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray img', img_gray)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 1)
    cv2.imshow('Thresholded', img_thresh)
    cv2.waitKey(0)

    img_v = cv2.dilate(img_thresh, np.ones((5, 5), np.uint8), iterations=1)
    img_v = cv2.erode(img_v, np.ones((9, 9), np.uint8), iterations=1)
    cv2.imshow('Result', img_v)
    cv2.waitKey(0)

    # cv2.imwrite('../hw_output/img2_res.jpg', img_v)


if __name__ == '__main__':
    # process_first_img('../Images for home task #1/hearts 1.png')
    process_second_img('/home/calvus/It-Jim course/les-01/Images for home task #1/hearts 2.png')
