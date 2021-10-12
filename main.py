import cv2
import numpy as np


def process_first_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 65, 1)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    img_v = cv2.dilate(img_thresh, np.ones((5, 5), np.uint8), iterations=1)
    img_v = cv2.erode(img_v, np.ones((9, 9), np.uint8), iterations=1)
    cv2.imshow('', img_v)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img1_res.jpg', img_v)


def process_second_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_gray, 5)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 1.9)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    img_v = cv2.erode(img_thresh, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('', img_v)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img2_res.jpg', img_v)


def process_third_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img[:, :, 0] = img[:, :, 1]
    img[:, :, 2] = img[:, :, 1]

    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 165, 0.5)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    img_v = cv2.erode(img_thresh, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('', img_v)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img3_res.jpg', img_v)


def process_fourth_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    (B, G, R) = cv2.split(img)
    (B, G, R) = (np.float32(B), np.float32(G), np.float32(R))
    diff = G - (B + R)
    cv2.imshow('', diff)
    cv2.waitKey(0)

    ran = cv2.inRange(diff, -80, 100)
    ran = 255 - ran
    cv2.imshow('', ran)
    cv2.waitKey(0)

    img_v = cv2.erode(ran, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('', img_v)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img4_res.jpg', img_v)


def process_fifth_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_dil = cv2.dilate(img_gray, np.ones((2, 2), np.uint8), iterations=5)
    cv2.imshow('', img_dil)
    cv2.waitKey(0)

    img_blur = cv2.GaussianBlur(img_dil, (15, 15), 3, borderType=cv2.BORDER_REPLICATE)
    cv2.imshow('', img_blur)
    cv2.waitKey(0)

    img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 175, 3)
    cv2.imshow('', img_bin)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img5_res.jpg', img_bin)


def process_sixth_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_bin = cv2.boxFilter(img_gray, 0, (7, 7), borderType=cv2.BORDER_REPLICATE)
    cv2.imshow('', img_bin)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_bin, 13)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    ret, img_thr = cv2.threshold(img_median, 115, 255, cv2.THRESH_BINARY)
    cv2.imshow('', img_thr)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img6_res.jpg', img_thr)


def process_seventh_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_dil = cv2.dilate(img_gray, np.ones((2, 2), np.uint8), iterations=4)
    cv2.imshow('', img_dil)
    cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(5, 5))
    img_op = clahe.apply(img_dil)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_op, 11)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_median, 13)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_median, 9)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    img_thr = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 205, 10)
    cv2.imshow('', img_thr)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img7_res.jpg', img_thr)


def process_eighth_img(img_name):

    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_dil = cv2.dilate(img_gray, np.ones((3, 3), np.uint8), iterations=3)
    cv2.imshow('', img_dil)
    cv2.waitKey(0)

    img_b = cv2.boxFilter(img_dil, 0, (5, 5), borderType=cv2.BORDER_REPLICATE)
    cv2.imshow('', img_b)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_b, 17)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(3, 3))
    img_op = clahe.apply(img_median)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_op, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 8)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img8_res.jpg', img_thresh)


def process_ninth_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_op = cv2.equalizeHist(img_gray)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img = cv2.bitwise_not(img_op)
    cv2.imshow('', img)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img9_res.jpg', img)


def process_tenth_img(img_name):
    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    img_h = cv2.dilate(img_gray, np.ones((1, 10), np.uint8), iterations=1)
    img_h = cv2.erode(img_h, np.ones((1, 10), np.uint8), iterations=1)
    cv2.imshow('', img_h)
    cv2.waitKey(0)

    img_r = cv2.erode(img_h, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imshow('', img_r)
    cv2.waitKey(0)

    cv2.imwrite('../hw_output/img10_res.jpg', img_r)


if __name__ == '__main__':
    process_first_img('../Images for home task #1/hearts 1.png')
    process_second_img('../Images for home task #1/hearts 2.png')
    process_third_img('../Images for home task #1/hearts 3.png')
    process_fourth_img('../Images for home task #1/hearts 4.png')
    process_fifth_img('../Images for home task #1/hearts 5.png')
    process_sixth_img('../Images for home task #1/hearts 6.png')
    process_seventh_img('../Images for home task #1/hearts 7.png')
    process_eighth_img('../Images for home task #1/hearts 8.png')
    process_ninth_img('../Images for home task #1/hearts 9.png')
    process_tenth_img('../Images for home task #1/hearts 10.png')

