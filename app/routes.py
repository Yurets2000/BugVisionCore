import base64
import io
import os
import shutil
import uuid

import cv2
import random

import numpy as np
from PIL import Image
from flask import current_app as app
from flask import request
from tensorflow import keras
import threading

import dataset_processing
import image_processing
import neural_networks.image_classifier
from config import Config
from dataset_processing import metadata_processor
from distutils.dir_util import copy_tree

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
CURR_DATASET_PATH = CURR_PATH + '/../dataset/' + Config.CURR_DATASET
CURR_SPLIT_DATASET_PATH = CURR_PATH + '/../dataset/' + Config.CURR_SPLIT_DATASET
CURR_METADATA_PATH = CURR_PATH + '/../dataset/' + Config.CURR_METADATA
CURR_MODEL_PATH = CURR_PATH + '/../neural_networks/' + Config.CURR_MODEL
TEMP_DATASET_PATH = CURR_PATH + '/../dataset/' + Config.TEMP_DATASET

meta_processor = metadata_processor.MetadataProcessor(CURR_METADATA_PATH)
model = keras.models.load_model(CURR_MODEL_PATH)

DATASET_MERGE_THRESHOLD = 0.002


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Bug Vision Core Module</h1>'''


@app.route('/classify', methods=['POST'])
def classify():
    request_body = request.json
    img_base64 = request_body['image'].split(',')[1]
    results_num = request_body['resultsNum']
    img = img_from_base64(img_base64)
    img = image_processing.resize_image_arr(img, dsize=(100, 100))
    img = np.array([img])
    predictions = model.predict(img)[0]
    predictions_results = get_prediction_results(predictions, results_num)

    return predictions_results


@app.route('/classes', methods=['GET'])
def classes():
    result = {
        'classes': meta_processor.get_column(1)
    }
    return result


@app.route('/temp-dataset-info', methods=['GET'])
def temp_dataset_info():
    result = {
        'samplesLoaded': dataset_processing.count_files_in_dir(TEMP_DATASET_PATH),
        'samplesNeededToNextUpdate': int(DATASET_MERGE_THRESHOLD * dataset_processing.count_files_in_dir(CURR_DATASET_PATH))
    }
    return result


@app.route('/load-image/<class_name>', methods=['GET'])
def load_image(class_name):
    if not (class_name in meta_processor.get_column(1)):
        raise ValueError('Value of \'class_name\' field should be one of the available classes')
    results = image_processing.search_by_tags(tags=[class_name], text='Insect', result_type='url', result_page=1,
                                              per_page=200)
    if len(results) > 0:
        random.shuffle(results)
        return results[0]
    return ''


@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Need to add new image to batch temp dir Check if number of images in temp dir < limit.
    # If number of images exceeds limit then preprocess them in new thread,
    # retrain model, save it and update in routes
    request_body = request.json
    if (not os.path.exists(TEMP_DATASET_PATH)):
        os.mkdir(TEMP_DATASET_PATH)
    img_base64 = request_body['image'].split(',')[1]
    class_name = request_body['className']
    class_label = meta_processor.get_field_by_key_field(1, class_name, 0)
    temp_class_path = '%s/class_%s' % (TEMP_DATASET_PATH, class_label)
    if (not os.path.exists(temp_class_path)):
        os.mkdir(temp_class_path)
    img = img_from_base64(img_base64)
    temp_file_path = '%s/%s_image_%s.jpg' % (temp_class_path, class_label, uuid.uuid4())
    cv2.imwrite(temp_file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    added_files_count = dataset_processing.count_files_in_dir(TEMP_DATASET_PATH)
    dataset_files_count = dataset_processing.count_files_in_dir(CURR_DATASET_PATH)
    if (added_files_count > int(dataset_files_count * DATASET_MERGE_THRESHOLD)):
        # Create new thread to preprocess, augment and merge datasets, then split dataset into new dataset
        thread = threading.Thread(target=update_dataset_and_retrain_model)
        thread.start()
    return ''


def img_from_base64(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def get_prediction_results(predictions, results_num):
    if (results_num > len(predictions)):
        raise ValueError('Value of \'results_num\' field should be less size of predictions array')
    result_indexes = np.argsort(-predictions)[:results_num]
    results = {}
    for result_index in result_indexes:
        row = meta_processor.get_row(result_index)
        results[row[1]] = round(predictions[result_index] * 100, 2)
    return results


def update_dataset_and_retrain_model():
    dataset_processing.preprocess_dataset(TEMP_DATASET_PATH, dsize=(100, 100), step_size=5)
    dataset_processing.augment_dataset(TEMP_DATASET_PATH)
    dataset_processing.merge_dataset(TEMP_DATASET_PATH, CURR_DATASET_PATH)
    shutil.rmtree(TEMP_DATASET_PATH)
    shutil.rmtree(CURR_SPLIT_DATASET_PATH)
    dir_items = os.listdir(CURR_DATASET_PATH)
    for dir_item in dir_items:
        copy_tree(os.path.join(CURR_DATASET_PATH, dir_item), os.path.join(CURR_SPLIT_DATASET_PATH, dir_item))
    dataset_processing.split_dataset(CURR_SPLIT_DATASET_PATH)

    neural_networks.image_classifier.train_model()
    global model
    model = keras.models.load_model(CURR_MODEL_PATH)
