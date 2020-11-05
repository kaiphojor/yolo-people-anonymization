from PIL import Image
import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_char1(img_):
    img_ori = img_
    #img_ori = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 11 --oem 0')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:", a)
    return a


def get_char2(img_):
    img_ori = img_
    #img_ori = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    chars = pytesseract.image_to_string(gray, lang='kor', config='--psm 11 --oem 0')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:",a)
    return a

def get_char3(img_):
    img_ori = img_
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 11 --oem 0')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('', chars)))
    #print(a)
    return a

def get_char4(img_):
    img_ori = img_
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 7 --oem 1')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:", a)
    return a


def get_char5(img_):
    img_ori = img_
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 12 --oem 1')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:", a)
    return a


def get_char6(img_):
    img_ori = img_
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 7 --oem 0')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:", a)
    return a

def get_char7(img_):
    img_ori = img_
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    chars = pytesseract.image_to_string(img_ori, lang='kor', config='--psm 12 --oem 0')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('',chars)))
    #print("ocr:", a)
    return a

def get_char8(img_):
    img_ori = img_
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    chars = pytesseract.image_to_string(gray, lang='kor', config='--psm 6 --oem 3')
    a = ((re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ|\r\n| ]+').sub('', chars)))
    print(a)
    #print("ocr:", a)
    return a


def check_name(text_, namelist):

    a = get_char1(text_)
    b = get_char2(text_)
    c = get_char3(text_)
    d = get_char4(text_)
    e = get_char5(text_)
    f = get_char6(text_)
    g = get_char7(text_)
    h = get_char8(text_)

    for name in namelist:
        if name in a:
            print("matched:", name, "ocr result: ",a)
            result = True
            break
        elif name in b:
            print("matched:", name, "ocr result: ",b)
            result = True
            break
        elif name in c:
            print("matched:", name, "ocr result: ",c)
            result = True
            break
        elif name in d:
            print("matched:", name, "ocr result: ",d)
            result = True
            break
        elif name in e:
            print("matched:", name, "ocr result: ",e)
            result = True
            break
        elif name in f:
            print("matched:", name, "ocr result: ",f)
            result = True
            break
        elif name in g:
            print("matched:", name, "ocr result: ",g)
            result = True
            break
        elif name in h:
            print("matched:", name, "ocr result: ",h)
            result = True
            break
        else:
            result = False
    return result,name

