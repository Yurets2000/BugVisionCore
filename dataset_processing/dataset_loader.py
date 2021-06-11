import csv
import os
import shutil

from image_processing import image_processor, flickr_processor


class DatasetLoader:
    def __init__(self, dir_path, metadata_file_path, samples_threshold):
        self.dir_path = dir_path
        self.metadata_file_path = metadata_file_path
        self.samples_threshold = samples_threshold

    def load_class(self, class_name, class_value):
        photo_urls = flickr_processor.search_by_tags(tags=[class_value], result_type='url')
        if len(photo_urls) > self.samples_threshold:
            full_dir_path = '%s/class_%s' % (self.dir_path, class_name)
            if os.path.exists(full_dir_path) and os.path.isdir(full_dir_path):
                shutil.rmtree(full_dir_path)
            os.mkdir(full_dir_path)
            for i in range(len(photo_urls)):
                file_name = '%s_image_%d.jpg' % (class_name, i)
                path = '%s/%s' % (full_dir_path, file_name)
                image_processor.download_image(photo_urls[i], path)

    def load_dataset(self):
        with open(self.metadata_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                class_name = row[0]
                class_value = row[1]
                self.load_class(class_name, class_value)
