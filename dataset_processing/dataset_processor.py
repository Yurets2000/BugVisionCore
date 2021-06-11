import os
import random
import shutil

from dataset_processing import data_augmentor
from image_processing import image_processor


def get_immediate_sub_dirs(path):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    result = list(filter(lambda item: os.path.isdir(path + '/' + item), os.listdir(path)))
    return result


def get_immediate_files(path):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    result = list(filter(lambda item: os.path.isfile(path + '/' + item), os.listdir(path)))
    return result


def count_files_in_dir(path):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    return sum([len(files) for r, d, files in os.walk(path)])


def num_to_str(pos):
    return chr(pos + 96)


def extract_class_name(path):
    index = path.rfind('_')
    return path[index + 1:]


def move_files(files, dir_path1, dir_path2):
    if not os.path.isdir(dir_path1):
        raise ValueError('Passed dir_path1 should refer to directory')
    if not os.path.isdir(dir_path2):
        raise ValueError('Passed dir_path2 should refer to directory')
    for i in range(len(files)):
        os.rename(dir_path1 + '/' + files[i], dir_path2 + '/' + files[i])


def sync_numerical_classes(path):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    sub_dirs = get_immediate_sub_dirs(path)
    for i in range(len(sub_dirs)):
        old_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(old_sub_dir_path)
        for j in range(len(files)):
            old_file_path = '%s/%s' % (old_sub_dir_path, files[j])
            new_file_path = '%s/%d_image_%d.jpg' % (old_sub_dir_path, i + 1, j + 1)
            os.rename(old_file_path, new_file_path)
        new_sub_dir_path = '%s/class_%d' % (path, i + 1)
        print(old_sub_dir_path, new_sub_dir_path)
        os.rename(old_sub_dir_path, new_sub_dir_path)


# WARNING: WORKS ONLY WHEN NUMBER OF CLASSES LESS THAN 26
def sync_alphabet_classes(path, temp=False):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    sub_dirs = get_immediate_sub_dirs(path)
    for i in range(len(sub_dirs)):
        old_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(old_sub_dir_path)
        for j in range(len(files)):
            old_file_path = '%s/%s' % (old_sub_dir_path, files[j])
            if temp:
                new_file_path = '%s/t_%s_image_%d.jpg' % (old_sub_dir_path, num_to_str(i + 1), j + 1)
            else:
                new_file_path = '%s/%s_image_%d.jpg' % (old_sub_dir_path, num_to_str(i + 1), j + 1)
            os.rename(old_file_path, new_file_path)
        if temp:
            new_sub_dir_path = '%s/t_class_%s' % (path, num_to_str(i + 1))
        else:
            new_sub_dir_path = '%s/class_%s' % (path, num_to_str(i + 1))
        print('\'%s\' -> \'%s\'' % (old_sub_dir_path, new_sub_dir_path))
        os.rename(old_sub_dir_path, new_sub_dir_path)


def switch_numerical_to_alphabet_classes(path):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    sub_dirs = get_immediate_sub_dirs(path)
    for i in range(len(sub_dirs)):
        old_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(old_sub_dir_path)
        for j in range(len(files)):
            old_file_path = '%s/%s' % (old_sub_dir_path, files[j])
            new_file_path = '%s/%s_image_%d.jpg' % (old_sub_dir_path,
                                                    num_to_str(int(extract_class_name(old_sub_dir_path))),
                                                    j + 1)
            os.rename(old_file_path, new_file_path)
        new_sub_dir_path = '%s/class_%s' % (path, num_to_str(int(extract_class_name(old_sub_dir_path))))
        print('\'%s\' -> \'%s\'' % (old_sub_dir_path, new_sub_dir_path))
        os.rename(old_sub_dir_path, new_sub_dir_path)


def resize_dataset(path, dsize=(300, 300)):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    sub_dirs = get_immediate_sub_dirs(path)
    for i in range(len(sub_dirs)):
        full_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(full_sub_dir_path)
        for j in range(len(files)):
            full_file_path = '%s/%s' % (full_sub_dir_path, files[j])
            print('Processing file \'%s\'...' % full_file_path)
            image_processor.resize_image(full_file_path, dsize=dsize)


def preprocess_dataset(path, dsize=(300, 300), step_size=1, padding=0.1):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    sub_dirs = get_immediate_sub_dirs(path)
    for i in range(len(sub_dirs)):
        full_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(full_sub_dir_path)
        for j in range(len(files)):
            full_file_path = '%s/%s' % (full_sub_dir_path, files[j])
            print('Processing file \'%s\'...' % full_file_path)
            image_processor.image_preprocess(full_file_path, dsize=dsize, step_size=step_size, padding=padding)


def split_dataset(path, test_percent=0.3):
    if test_percent >= 1 or test_percent <= 0:
        raise ValueError('Value of \'test_percent\' field should be between 0 and 1')
    sub_dirs = get_immediate_sub_dirs(path)
    train_path = path + '/train'
    os.mkdir(train_path)
    test_path = path + '/test'
    os.mkdir(test_path)
    for i in range(len(sub_dirs)):
        full_sub_dir_path = '%s/%s' % (path, sub_dirs[i])
        files = get_immediate_files(full_sub_dir_path)
        test_num = int(len(files) * test_percent)
        random.shuffle(files)
        test_files = files[0:test_num]
        test_sub_dir_path = test_path + '/' + sub_dirs[i]
        os.mkdir(test_sub_dir_path)
        move_files(test_files, full_sub_dir_path, test_sub_dir_path)
        train_files = files[test_num:len(files)]
        train_sub_dir_path = train_path + '/' + sub_dirs[i]
        os.mkdir(train_sub_dir_path)
        move_files(train_files, full_sub_dir_path, train_sub_dir_path)
        shutil.rmtree(full_sub_dir_path)


def augment_dataset(path, rot=True, v_flip=True, h_flip=False):
    if not os.path.isdir(path):
        raise ValueError('Passed path should refer to directory')
    for root, _, files in os.walk(path):
        for file in files:
            print('Augmentation of file \'%s\'' % file)
            augmentor = data_augmentor.DataAugmentor(root, file)
            augmentor.image_augment(rot=rot, v_flip=v_flip, h_flip=h_flip)


def merge_datasets(from_paths, to_path):
    if not os.path.isdir(to_path):
        raise ValueError('Value of \'to_path\' field not refer to directory')
    for from_path in from_paths:
        if not os.path.isdir(from_path):
            raise ValueError('One of the \'from_paths\' elements not refer to directory')
    for from_path in from_paths:
        merge_dataset(from_path, to_path)


def merge_dataset(from_path, to_path):
    if not os.path.isdir(from_path):
        raise ValueError('Value of \'from_path\' field not refer to directory')
    if not os.path.isdir(to_path):
        raise ValueError('Value of \'to_path\' field not refer to directory')
    to_path_sub_dirs = get_immediate_sub_dirs(to_path)
    for to_path_sub_dir in to_path_sub_dirs:
        full_from_path_sub_dir = from_path + '/' + to_path_sub_dir
        full_to_path_sub_dir = to_path + '/' + to_path_sub_dir
        if (os.path.exists(full_from_path_sub_dir)):
            files = get_immediate_files(full_from_path_sub_dir)
            move_files(files, full_from_path_sub_dir, full_to_path_sub_dir)
    sync_alphabet_classes(to_path, temp=True)
    sync_alphabet_classes(to_path, temp=False)


if __name__ == "__main__":
    # switch_numerical_to_alphabet_classes('v1')
    # augment_dataset('../dataset/v6')
    # preprocess_dataset('v2', dsize=(250, 250), step_size=5)
    # sync_alphabet_classes('../dataset/v6', temp=True)
    # sync_alphabet_classes('../dataset/v6', temp=False)
    resize_dataset('../dataset/v8', dsize=(75, 75))
    split_dataset('../dataset/v8', test_percent=0.3)
    resize_dataset('../dataset/v9', dsize=(125, 125))
    split_dataset('../dataset/v9', test_percent=0.3)
