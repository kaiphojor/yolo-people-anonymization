import numpy as np
import random
from Cryptodome.Cipher import AES
from Cryptodome import Random
import cv2


class Encrypt:
    def __init__(self,image, image_path, key):
        self.image_path = image_path
        self.key = key
        self.image = image


    def convert(self):
        image = self.image
        image = cv2.imencode(".jpg", image)[1].tobytes()
        image = bytearray(image)

        for index, value in enumerate(image):
            image[index] = value ^ self.key
        fo = open(self.image_path, "wb")
        fo.write(image)
        fo.close()


class Decrypt:
    def __init__(self, image_path, key, img_shape):
        self.image_path = image_path
        self.key = key
        self.img_shape = img_shape


    def convert(self):
        fo = open(self.image_path, "rb")
        shape = self.img_shape

        image = fo.read()
        fo.close()
        image = bytearray(image)
        for index, value in enumerate(image):
            image[index] = value ^ self.key

        image = bytes(image)
        #print("image_dec:", type(image), image)
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        #print("decoded: ",decoded,len(decoded),decoded.shape,shape)
        return decoded

