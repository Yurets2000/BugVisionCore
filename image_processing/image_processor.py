import cv2
import requests

import image_saliency as psal


def download_image(url, path):
    response = requests.get(url)
    file = open(path, "wb")
    file.write(response.content)
    file.close()


def resize_image_arr(img, dsize, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(img, dsize=dsize, interpolation=interpolation)


def resize_image(path, dsize, interpolation=cv2.INTER_CUBIC):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, dsize=dsize, interpolation=interpolation)
    cv2.imwrite(path, resized_img)


def crop_image(path, step_size=1, padding=0.1):
    if padding >= 1 or padding < 0:
        raise ValueError('Value of \'padding\' field should between 0 and 1')
    img = cv2.imread(path)
    mbd = psal.get_saliency_mbd(path).astype('uint8')
    binary_sal = psal.binarise_saliency_map(mbd, method='adaptive')
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    height = len(binary_sal)
    width = len(binary_sal[0])
    if step_size >= min(width, height):
        raise ValueError('Value of \'step_size\' field should be less than min(width, height)')
    for i in range(0, height, step_size):
        for j in range(0, width, step_size):
            if (i > step_size) and not binary_sal[i - step_size][j] and binary_sal[i][j] and y1 == 0:
                y1 = i
            if (i < height - step_size - 1) and not binary_sal[i + step_size][j] and binary_sal[i][j]:
                y2 = i
    for j in range(0, width, step_size):
        for i in range(0, height, step_size):
            if (j > step_size) and not binary_sal[i][j - step_size] and binary_sal[i][j] and x1 == 0:
                x1 = j
            if (j < width - step_size - 1) and not binary_sal[i][j + step_size] and binary_sal[i][j]:
                x2 = j
    y1 = max(y1 - int(height * padding), 0)
    y2 = min(y2 + int(height * padding), height)
    x1 = max(x1 - int(width * padding), 0)
    x2 = min(x2 + int(width * padding), width)
    cropped_img = img[y1:y2, x1:x2]
    cv2.imwrite(path, cropped_img)


def image_preprocess(path, dsize=(300, 300), step_size=1, padding=0.1):
    try:
        crop_image(path, step_size=step_size, padding=padding)
    except Exception as ex:
        print('Exception occur during image_processing cropping', ex)
    resize_image(path, dsize=dsize)
