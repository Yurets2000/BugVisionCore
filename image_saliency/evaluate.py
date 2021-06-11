import os
import cv2
import image_saliency


def evaluate(img_dir, gt_dir, methods):
    results_precision = {}
    results_recall = {}
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            basename = os.path.splitext(filename)[0]
            if os.path.isfile(gt_dir + '/' + basename + '.png'):
                gt_image_path = gt_dir + '/' + basename + '.png'
            elif os.path.isfile(gt_dir + '/' + basename + '.jpg'):
                gt_image_path = gt_dir + '/' + basename + '.jpg'
            else:
                print('No match in gt directory for file' + str(filename) + ', skipping.')
                continue
            print(img_dir + '/' + filename)
            sal_image = image_saliency.get_saliency_mbd(img_dir + '/' + filename).astype('uint8')
            gt_image = cv2.imread(gt_image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            cv2.imshow('sal', sal_image)
            cv2.imshow('img', gt_image)
            cv2.waitKey(0)
            if gt_image.shape != sal_image.shape:
                print('Size of image_processing and GT image_processing does not match, skipping')
                continue
    return (results_precision, results_recall)
