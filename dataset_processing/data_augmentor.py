import cv2


class DataAugmentor:
    def __init__(self, path, image_name):
        self.path = path
        self.name = image_name
        self.image = cv2.imread(path + '/' + image_name)

    def __rotate(self, image, angle=90, scale=1.0):
        w = image.shape[1]
        h = image.shape[0]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        image = cv2.warpAffine(image, m, (w, h))
        return image

    def __flip(self, image, v_flip=False, h_flip=False):
        if h_flip or v_flip:
            if h_flip and v_flip:
                c = -1
            else:
                c = 0 if v_flip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def image_augment(self, rot=True, v_flip=True, h_flip=False):
        img = self.image.copy()
        name_int = self.name[:len(self.name) - 4]
        if v_flip or h_flip:
            img_flip = self.__flip(img, v_flip=v_flip, h_flip=h_flip)
            cv2.imwrite('%s/%s_flip.jpg' % (self.path, str(name_int)), img_flip)
        if rot:
            img_rot = self.__rotate(img)
            cv2.imwrite('%s/%s_rot.jpg' % (self.path, str(name_int)), img_rot)
