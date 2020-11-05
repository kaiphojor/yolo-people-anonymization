import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, removeAllFile)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import ocr
import secure as ied


def detect(opt,save_img=False):
    out, source, weights, imgsz, namelist = \
        opt.output, opt.source, opt.weights, opt.img_size, opt.namelist

    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) #if device.type != 'cpu' else None  # run once
    idx=0
    ckname = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        idx+=1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        cnt = 0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                cntname=0
                # Write results
                img2 = im0.copy()
                nperson=[]
                nname=[]
                for *xyxy, conf, cls in reversed(det):

                    if save_img :  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                ########################################################################################################
                        ##classes 변수 생성 (이름)
                        classes = names[int(cls)]
                        ##classes 변수 함수에 추가
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3, classes=classes)
                        ##사람이라고 판단한 물체의 각 좌표 리스트에 저장
                        if classes == 'person':
                            nperson.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                        if classes == 'name':
                            nname.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                ##이름 리스트의 크기가 0보다 클 때 미리 복사해둔 프레임의 구역으로 이미지 덮기
                #print(len(nperson))
                if len(nname) >0:
                    key = 45
                    for pi in range(len(nperson)):
                        check = False
                        for ii in range(len(nname)):
                            if nname[ii][1]>=nperson[pi][1] and nname[ii][3]<=nperson[pi][3] and nname[ii][0]>=nperson[pi][0] \
                                    and nname[ii][2]<=nperson[pi][2] and check==False:
                                check = True
                                proi=img2[nname[ii][1]:nname[ii][3],nname[ii][0]:nname[ii][2]]
                                temp_img = "{0}_{1}_{2}_{3}.jpg".format(nname[ii][1],nname[ii][3],nname[ii][0],nname[ii][2])
                                image_path = "./temp/{0}".format(temp_img)
                                img_shape = proi.shape
                            # print(proi)
                                #image_path2 = "./temp/tt_{0}".format(temp_img)

                        #######################################
                                encrypt_function(proi,image_path, key)
                        # os.remove(image_path)
                                text_ = decrypt_function(image_path,key,img_shape)
                                #cv2.imwrite(image_path2, text_)
                        #########################################
                        #print("coord:",nname[ii][1],nname[ii][3],nname[ii][0],nname[ii][2])

                        # OCR (이름 매칭 확인) => return True / False
                                result,check_name = ocr.check_name(text_,namelist)
                                if result ==True:
                                    cntname+=1
                                    if check_name not in ckname:
                                        ckname.append(check_name)
                                    roi = img2[nperson[pi][1]:nperson[pi][3], nperson[pi][0]:nperson[pi][2]]
                                    im0[nperson[pi][1]:nperson[pi][3], nperson[pi][0]:nperson[pi][2]] = roi
                #cv2.imwrite('.\check\{}.jpg'.format(idx),im0)
                ########################################################################################################
            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            removeAllFile('./temp')
            # Save results (image with detections)
            if save_img:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter('./output.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)


def encrypt_function(image, image_path, key):
    enc = ied.Encrypt(image, image_path, key)
    enc.convert()
    #print("Encrypted")


def decrypt_function(image_path, key, img_shape):
    dec = ied.Decrypt(image_path, key,img_shape)
    decoded = dec.convert()
    #print("Decrypted")
    return decoded





def main(text_list,video_path):
    print("--in main--")
    print(text_list)
    print(video_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='person600_deleted_epoch20.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=video_path, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='output confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--namelist', default=text_list, help='update all models')
    opt = parser.parse_args()
    print(opt)

    detect(opt)
