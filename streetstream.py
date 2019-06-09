import sys
import cv2 as cv
import numpy as np
from collections import namedtuple

WIDTH = 1920
HEIGHT = 1080

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

classesFile = "model/coco.names"
modelConfiguration = "model/yolov3.cfg"
modelWeights = "model/yolov3.weights"


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, classes, classId, conf, left, top, right, bottom, color):
    cv.rectangle(frame, (left, top), (right, bottom), color, 3) # (255, 178, 50)

    #label = '%.2f' % conf

    #if classes:
        #assert (classId < len(classes))
        #label = '%s:%s' % (classes[classId], label)

    #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 #(255, 255, 255), cv.FILLED)
    #cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, classes, masks, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                object_center = [center_y, center_x]
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                centers.append(object_center)

    #centers = centers
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if centers[i] not in masks.gray:
            if classes[classIds[i]] == 'person':
                if (centers[i] not in masks.ped) and (centers[i] not in masks.crs):
                    drawPred(
                        frame, classes, classIds[i], confidences[i],
                        left, top, left + width, top + height, (255, 0, 0))
            elif classes[classIds[i]] in ['car', 'truck']:
                if centers[i] not in masks.vhc:
                    drawPred(
                        frame, classes, classIds[i], confidences[i],
                        left, top, left + width, top + height, (255, 0, 0))


def handle_image(image, i):
    font = cv.FONT_HERSHEY_SIMPLEX
    image = cv.putText(image, str(i), (10, 200), font, 4, (0, 0, 255), 2, cv.LINE_AA)
    return image


def handle_image_with_detection(frame, classes, masks, net, i):
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, classes, masks, outs)
    return frame


def main():
    i = 0

    ped_mask = cv.imread('masks/ped.jpg')[:, :, 0]
    vhc_mask = cv.imread('masks/vhc.jpg')[:, :, 0]
    crs_mask = cv.imread('masks/crs.jpg')[:, :, 0]
    gray_mask = cv.imread('masks/gray.jpg')[:, :, 0]
    MaskSet = namedtuple('MaskSet', ['ped', 'vhc', 'crs', 'gray']) # , 'ped', 'vhc', nda

    # red - pedestrian 0xf34423 [243, 68, 35]
    # gray - non detectable area 0x5c5c5c [92, 92, 92]
    # blue - vehicles 0x0066fa [0, 102, 250]
    # yellow - pedestrian crossing 0xffee3d [255, 253, 0]

    # pedestrian crossing 247
    # ped 130
    # gray 92
    # vhc 70

    masks = MaskSet(
        ped=np.argwhere(ped_mask > 0).tolist(),
        vhc=np.argwhere(vhc_mask > 0).tolist(),
        crs=np.argwhere(crs_mask > 0).tolist(),
        gray=np.argwhere(gray_mask > 0).tolist()
    )
    '''
    nda=np.argwhere(mask == [92, 92, 92])[:, :2].tolist(),
        ped=np.argwhere(mask == [255, 0, 0])[:, :2].tolist(),
        vhc=np.argwhere(mask == [0, 0, 255])[:, :2].tolist()
    '''
    '''
    #nda=np.unique(np.argwhere(mask == [92, 92, 92])[:, :2], axis=0).tolist(),
        #ped=np.unique(np.argwhere(mask == [255, 0, 0])[:, :2], axis=0).tolist(),
        #vhc=np.unique(np.argwhere(mask == [0, 0, 255])[:, :2], axis=0).tolist()
    '''

    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) # CPU

    sys.stdin = sys.stdin.detach()  # now it's binary stream
    sys.stdout = sys.stdout.detach()  # now it's binary stream
    while True:
        raw_data = sys.stdin.read(WIDTH*HEIGHT*3)

        image = np.frombuffer(raw_data, dtype='uint8').reshape((HEIGHT, WIDTH, 3)) #.astype('float32') #/ 255

        #result = handle_image(image, i)
        result = handle_image_with_detection(image, classes, masks, net, i)

        result_bytes = result.tobytes() # astype('uint8').

        sys.stdout.write(result_bytes)  # result_bytes raw_data

        i += 1


if __name__ == '__main__':
    main()
