import numpy as np
import cv2
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
import sounddevice as sd
from piper.voice import PiperVoice
import time

HandPth="hand.hef"
YoloPth="yolov11n.hef"
model = "norman.onnx"
voice = PiperVoice.load(model)

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
    image=cv2.resize(image,(640,640))
    return image


if __name__ == '__main__':
    handDet = Hailo(HandPth)
    yoloDet = Hailo(YoloPth)
    print(yoloDet.get_input_shape())
    model_h, model_w, _ = yoloDet.get_input_shape() # Assuming get_input_shape returns (height, width)

    video_w, video_h = 1280, 960 # These are the main stream dimensions as used by extract_detections
    
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
    # lores_w will be adjusted to match video_w/video_h aspect ratio, but height will be model_h
    lores_w = int(round(model_w * (video_w / video_h))) # This might be model_w, not lores_w
    lores = {'size': (lores_w, model_h), 'format': 'BGR888'} # This is the actual size of 'lores' frames
    crop_ratio = model_w / lores_w # This ratio is used to adjust x-coordinates in extract_detections

    controls = {'FrameRate': 30}
    config = piCam2.create_preview_configuration(main, lores=lores, controls=controls)
    piCam2.configure(config)

    #piCam2.start_preview(Preview.QT, x=0, y=0, width=video_w, height=video_h)
    piCam2.start()

    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()
    rest=1
    t_start=time.time()
    while True:
        try:
            frame = piCam2.capture_array('lores') # This frame has dimensions (model_h, lores_w, 3)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cropped_frame = crop_to_square(frame) # This frame has dimensions (model_h, model_h, 3) if model_h is min
            #cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR) # Your original comment suggests this was RGB. Ensure it's BGR for OpenCV.
            frame=cropped_frame
            frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
            if (time.time()-t_start)>=rest:
                # Run inference on the preprocessed frame
                results = yoloDet.run(cropped_frame)
                hnResults = handDet.run(cropped_frame)

                # Extract detections from the inference results.
                # Note: extract_detections scales bounding boxes to video_w, video_h
                for c in range(5):
                    detections = extract_detections(results, video_w, video_h, class_names,0.50, crop_ratio)
                    handDetections= extract_detections(hnResults, video_w, video_h, ["hand"],0.37, crop_ratio)

                # --- Draw detections on the frame ---
                # The frame is 'lores' size (lores_w, model_h).
                # The detections are scaled to (video_w, video_h) by extract_detections.
                # We need to scale them down to the frame's size (lores_w, model_h).
                scale_x_to_display =  lores_w / video_w
                scale_y_to_display = model_h / video_h

                for name, bbox, score in detections:
                    x0, y0, x1, y1 = bbox
                    # Scale coordinates to the frame's actual dimensions
                    x0_draw = int(x0 * scale_x_to_display)
                    y0_draw = int(y0 * scale_y_to_display)
                    x1_draw = int(x1 * scale_x_to_display)
                    y1_draw = int(y1 * scale_y_to_display)

                    color = (0, 255, 0) # Green for YOLO detections (BGR format)
                    thickness = 1
                    if name != "person":
                        cv2.rectangle(frame, (x0_draw, y0_draw), (x1_draw, y1_draw), color, thickness)
                        cv2.putText(frame, f"{name}: {score:.2f}", (x0_draw, y0_draw - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                for name, bbox, score in handDetections:
                    x0, y0, x1, y1 = bbox
                    # Scale coordinates to the frame's actual dimensions
                    x0_draw = int(x0 * scale_x_to_display)
                    y0_draw = int(y0 * scale_y_to_display)
                    x1_draw = int(x1 * scale_x_to_display)
                    y1_draw = int(y1 * scale_y_to_display)

                    color = (255, 0, 0) # Blue for hand detections (BGR format)
                    thickness = 1
                    cv2.rectangle(frame, (x0_draw, y0_draw), (x1_draw, y1_draw), color, thickness)
                    cv2.putText(frame, f"{name}: {score:.2f}", (x0_draw, y0_draw + 20), # Offset text to avoid overlap
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                # Using these detections we proceed to generate the message
                def getCenterPoint(bbox):
                    # Remember bbox is already scaled to video_w, video_h
                    # The center point should also be scaled to video_w, video_h for consistency with bbox
                    cpx = bbox[0] + (bbox[2] - bbox[0]) / 2
                    cpy = bbox[1] + (bbox[3] - bbox[1]) / 2
                    return [int(cpx), int(cpy)]

                lHandCp=[-1,-1]
                rHandCp=[-1,-1]
                curCp=[-1,-1]
                curScore=0.0
                for name,bbox,score in handDetections:
                    curCp=getCenterPoint(bbox)
                    if curCp[0]<int(video_w/2): # Check against the main video width
                        if score>curScore:
                            lHandCp=curCp
                    else:
                        if score>curScore:
                            rHandCp=curCp
                
                closestObj_l=["",[0,0]] # [ Name, [ delX , delY ] ] if delX pos means move hand right; if delY pos means move hand up
                closestObj_r=["",[0,0]] # [ Name, [ delX , delY ] ] if delX pos means move hand right; if delY pos means move hand up
                closestObj_u=["",[0,0]] # closest object to the user if delX pos move head right if delY pos move up and vice versa for all
                
                # Calculate distances relative to the center of the hand/user in the full video_w, video_h space
                curCp=[-1,-1]
                delX=0
                delY=0
                if lHandCp != [-1, -1] or rHandCp != [-1, -1]:
                    for name,bbox,score in detections:
                        if name!="person":
                            curCp=getCenterPoint(bbox)
                            if lHandCp != [-1,-1]: # If left hand is detected
                                print(name)
                                current_delX_l = curCp[0] - lHandCp[0]
                                current_delY_l = curCp[1] - lHandCp[1]
                                if closestObj_l[1] == [0,0] or (abs(current_delX_l) + abs(current_delY_l)) < (abs(closestObj_l[1][0]) + abs(closestObj_l[1][1])):
                                    closestObj_l=[ name , [ current_delX_l, current_delY_l ] ]
                            
                            if rHandCp != [-1,-1]: # If right hand is detected
                                current_delX_r = curCp[0] - rHandCp[0]
                                current_delY_r = curCp[1] - rHandCp[1]
                                if closestObj_r[1] == [0,0] or (abs(current_delX_r) + abs(current_delY_r)) < (abs(closestObj_r[1][0]) + abs(closestObj_r[1][1])):
                                    closestObj_r=[ name , [ current_delX_r, current_delY_r ] ]
                else: # No hands detected, find closest object to center of user's view
                    min_dist_to_center = float('inf')
                    for name, bbox, score in detections:
                        if name != "person":
                            curCp = getCenterPoint(bbox)
                            current_delX = curCp[0] - int(video_w / 2)
                            current_delY = curCp[1] - int(video_h / 2)
                            dist_to_center = (current_delX**2 + current_delY**2)**0.5
                            
                            if dist_to_center < min_dist_to_center:
                                min_dist_to_center = dist_to_center
                                closestObj_u = [name, [current_delX, current_delY]]


                messages=["","",""]
                if closestObj_l != ["",[0,0]] or closestObj_r != ["",[0,0]]: # If any hand is detected
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
                else: # No hands detected, use closest object to user's view center
                    if closestObj_u != ["",[0,0]]:
                        delX,delY=closestObj_u[1]
                        x=""
                        y=""
                        messages[2]="move your head slightly to your "
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
                
                # Display messages on the console (as before)
                print(messages[0] + "\n" + messages[1] + "\n" + messages[2])
                for message in messages:
                    if message != "":
                        for audio_bytes in voice.synthesize_stream_raw(message):
                            int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                            stream.write(int_data)
                        #time.sleep(1)
                t_start=time.time()
            
            # --- End of while loop changes for cv2.imshow ---
            # Wait for 1 millisecond. If 'q' is pressed, break the loop.
        except Exception as e: 
            # --- After the while loop, add cleanup ---
            print(e)
            cv2.destroyAllWindows() # Close all OpenCV windows
            piCam2.stop() # Stop the Picamera2 stream
            handDet.close() # Close Hailo models
            yoloDet.close() # Close Hailo models
            stream.stop()
            stream.close()
