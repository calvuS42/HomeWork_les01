import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(13, 13))
    img_op = clahe.apply(img_gray)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_op, 15)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    img_dil = cv2.dilate(img_gray, np.ones((3, 3), np.uint8), iterations=3)
    cv2.imshow('', img_dil)
    cv2.waitKey(0)

    img_b = cv2.boxFilter(img_dil, 0, (7, 7), borderType=cv2.BORDER_REPLICATE)
    cv2.imshow('', img_b)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_b, 17)
    cv2.imshow('', img_median)
    cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(3, 3))
    img_op = clahe.apply(img_median)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img_thresh = cv2.adaptiveThreshold(img_op, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 5)
    cv2.imshow('', img_thresh)
    cv2.waitKey(0)

    # _, img_thr = cv2.threshold(img_op, 115, 255, cv2.THRESH_BINARY)
    # cv2.imshow('', img_thr)
    # cv2.waitKey(0)


def process_eighth_img(img_name):

    img = cv2.imread(img_name)
    cv2.imshow('', img)
    cv2.waitKey(0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', img_gray)
    cv2.waitKey(0)

    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    # img_op = clahe.apply(img_gray)
    img_op = cv2.equalizeHist(img_gray)
    cv2.imshow('', img_op)
    cv2.waitKey(0)

    img_median = cv2.medianBlur(img_op, 5)
    cv2.imshow('', img_median)
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

    cv2.imwrite('../hw_output/img10_res.jpg', img_thresh)


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

    cv2.imwrite('../hw_output/img10_res.jpg', img)


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


def show_img_by_channels(img):
    """
    Function which draws all channels as a wider single-channel image.
    :return: No return
    """

    # Create an empty image (numpy array) for wide image result
    img_split_ch = np.zeros((img.shape[0], img.shape[1] * img.shape[2]), dtype=np.uint8)

    # Filling the empty image with channels intensities
    for i in range(img.shape[2]):
        img_split_ch[:, img.shape[1] * i: img.shape[1] * (i + 1)] = img[:, :, i]

    # Show the channel-separated image
    cv2.imshow('Three channels of color image in its color space', img_split_ch)

    # Waiting for a key (milliseconds to wait or 0 for infinite wait for a key stroke)
    cv2.waitKey(0)  # won't draw anything without this function!!!


if __name__ == '__main__':
    # process_first_img('../Images for home task #1/hearts 1.png')
    # process_second_img('../Images for home task #1/hearts 2.png')
    # process_third_img('../Images for home task #1/hearts 3.png')
    # process_fourth_img('../Images for home task #1/hearts 4.png')
    # process_fifth_img('../Images for home task #1/hearts 5.png')
    # process_sixth_img('../Images for home task #1/hearts 6.png')
    process_seventh_img('../Images for home task #1/hearts 7.png')
    # process_eighth_img('../Images for home task #1/hearts 8.png')
    # process_ninth_img('../Images for home task #1/hearts 9.png')
    # process_tenth_img('../Images for home task #1/hearts 10.png')

