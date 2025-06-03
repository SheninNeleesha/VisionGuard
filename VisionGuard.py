import numpy as np
import cv2
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo

HandPth="/home/pi/vg/hand.hef"
YoloPth="/home/pi/vg/yolo11n.hef"

def extract_detections(hailo_output, w, h, class_names, threshold, crop_ratio):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                x0_crop = (crop_ratio * (x0 - 0.5) + 0.5)
                x1_crop = (crop_ratio * (x1 - 0.5) + 0.5)

                bbox = (int(x0_crop * w), int(y0 * h), int(x1_crop * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results


def crop_to_square(image):
    # Crop the image to be square
    h, w, _ = image.shape

    if not h == w:
        if w > h:
            w_split = (w - h) // 2
            image = np.ascontiguousarray(image[:, w_split:w_split + h])
        else:
            h_split = (h - w) // 2
            image = np.ascontiguousarray(image[:, :, h_split:h_split + w])
    return image


if __name__ == '__main__':
    handDet = Hailo(HandPth)
    yoloDet = Hailo(YoloPth)
    model_h, model_w, _ = yoloDet.get_input_shape()
    video_w, video_h = 1280, 960
    # The yolo11n classes
    class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']
    detections = None
    piCam2 = Picamera2()
    main = {'size': (video_w, video_h), 'format': 'XRGB8888'}

    # Keep the aspect ratio of the main feed
    lores_w = int(round(model_w * (video_w / video_h)))
    lores = {'size': (lores_w, model_h), 'format': 'BGR888'}
    crop_ratio = model_w / lores_w

    controls = {'FrameRate': 30}
    config = piCam2.create_preview_configuration(main, lores=lores, controls=controls)
    piCam2.configure(config)

    piCam2.start_preview(Preview.QT, x=0, y=0, width=video_w, height=video_h)
    piCam2.start()
    while True:
        frame = picam2.capture_array('lores')
        cropped_frame = crop_to_square(frame)
        # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)

        # Run inference on the preprocessed frame
        results = yoloDet.run(cropped_frame)
        hnResults = handDet.run(cropped_frame)

        # Extract detections from the inference results
        detections = extract_detections(results, video_w, video_h, class_names,0.50, crop_ratio)
        handDetections= extract_detections(hnResults, video_w, video_h, ["hand"],0.37, crop_ratio)
        # Using these detections we proceed to generate the message
        # bbox = (int(x0_crop * w), int(y0 * h), int(x1_crop * w), int(y1 * h))
        def getCenterPoint(bbox):
            cpx=(bbox[2]-bbox[0])/2
            cpy=(bbox[3]-bbox[1])/2
            cp=[int(cpx),int(cpy)]
            return cp
        # [class_names[class_id], bbox, score]
        lHandCp=[-1,-1]
        rHandCp=[-1,-1]
        curCp=[-1,-1]
        curScore=0.0
        for name,bbox,score in handDetections:
            curCp=getCenterPoint(bbox)
            if curCp[0]<int(video_w/2):
                if score>curScore:
                    lHandCp=curCp
            else:
                if score>curScore:
                    rHandCp=curCp
        closestObj_l=["",[0,0]] # [ Name, [ delX , delY ] ] if delX pos means move hand right; if delY pos means move hand up
        closestObj_r=["",[0,0]] # [ Name, [ delX , delY ] ] if delX pos means move hand right; if delY pos means move hand up
        closestObj_u=["",[0,0]] # closest object to the user if delX pos move head right if delY pos move up and vice versa for all
        curCp=[-1,-1]
        delX=0
        delY=0
        if lHandCp != [-1, -1] or rHandCp != [-1, -1]:
            for name,bbox,score in detections:
                curCp=getCenterPoint(bbox)
                if curCp[0]<int(video_w/2):
                    if lHandCp!=[-1,-1]:
                        delX=curCp[0]-lHandCp[0]
                        delY=curCp[1]-lHandCp[1]
                        closestObj_l=[ name , [ delX, delY ] ]
                    else:
                        if rHandCp!=[-1,-1]:
                            delX=curCp[0]-rHandCp[0]
                            delY=curCp[1]-rHandCp[1]
                            closestObj_r=[ name , [ delX, delY ] ]
        else:
            for name, bbox, score in detections:
                if (curCp[0]-int(video_w/2))^2<delX^2 or (curCp[1]-int(video_h/2))^2<delY^2:
                    delX=curCp[0]-int(video_w/2)
                    delY=curCp[1]-int(video_h/2)
                    closestObj_u=[ name , [ delX, delY ] ]
        messages=["","",""]
        if closestObj_u == ["",[0,0]]:
            if closestObj_l != ["",[0,0]]:
                delX,delY=closestObj_l[1]
                x=""
                y=""
                messages[0]="move left hand slightly to your "
                if delX>0 and delY>0:
                    x="right"
                    y="up"
                elif delX>0 and delY<0:
                    x="right"
                    y="down"
                elif delX<0 and delY>0:
                    x="left"
                    y="up"
                else:
                    x="left"
                    y="down"
            messages[0] += f'{x} then {y} to reach {closestObj_l[0]}'
            if closestObj_r != ["",[0,0]]:
                delX,delY=closestObj_r[1]
                x=""
                y=""
                messages[1]="move right hand slightly to your "
                if delX>0 and delY>0:
                    x="right"
                    y="up"
                elif delX>0 and delY<0:
                    x="right"
                    y="down"
                elif delX<0 and delY>0:
                    x="left"
                    y="up"
                else:
                    x="left"
                    y="down"
                messages[1] += f'{x} then {y} to reach {closestObj_r[0]}'
        else:
            if closestObj_u != ["",[0,0]]:
                delX,delY=closestObj_u[2]
                x=""
                y=""
                messages[2]="tilt your head slightly to your "
                if delX>0 and delY>0:
                    x="right"
                    y="up"
                elif delX>0 and delY<0:
                    x="right"
                    y="down"
                elif delX<0 and delY>0:
                    x="left"
                    y="up"
                else:
                    x="left"
                    y="down"
                messages[2] += f'{x} then {y} to reach {closestObj_u[0]}'