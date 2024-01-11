#!/usr/bin/env python3.8
 
import numpy as np 
import cv2 
def crop(x1,y1,x2,y2,ids,size_x=448,size_y=448,h=1080,width=2400):
    xc=x1+(x2-x1)/2
    yc=y1+(y2-y1)/2
    x1=xc-size_x/2
    x2=xc+size_x/2
    y1=yc-size_y/2
    y2=yc+size_y/2
    iters=x1.shape[0]
    for i in range(iters):
        x11,x22,y11,y22=x1[i],x2[i],y1[i],y2[i]
        if x11<0:
            x22=x22+(0-x11)
            x11=0
        if x22>width:
            x11=x11-(x22-width)
            x22=width
        if y11<0:
            y22=y22+(0-y11)
            y11=0
        if y22>width:
            y11=y11-(y22-h)
            y22=h
        x1[i],x2[i],y1[i],y2[i]=int(x11),int(y11),int(x22),int(y22)

    return x1,x2,y1,y2,ids

def classify_crop_img(img,bbx,model_cls,**kwargs):
            bbx = bbx[np.argsort((bbx[:, 2] - bbx[:, 0])/2+bbx[:, 0])]
            x1,y1,x2,y2,ids=bbx[...,:5].T
            img_crops=np.array(crop(x1,y1,x2,y2,ids)).T.astype(np.int32)
            img_crops_agg={'id':[],'severity':[],"probablity":[]}
            for i in range(img_crops.shape[0]):
                x1,y1,x2,y2,id=img_crops[i]
                result=model_cls(img[y1:y2,x1:x2,:],stream=False,**kwargs)
                probablity=round(result[0].probs.data.max().item(),2)
                class_prediction=result[0].names[result[0].probs.top1]

                img_crops_agg['severity'].append(class_prediction)
                img_crops_agg['probablity'].append(probablity)
                img_crops_agg['id'].append(id)

            return img_crops_agg

def publish_bbx(message,bbx):
                    x1=[]
                    y1=[]
                    x2=[]
                    y2=[]
                    confs=[]
                    ids=[]
                    clss=[]

                    Output_message=message()
                    x1.extend(bbx[...,0].tolist())
                    y1.extend(bbx[...,1].tolist())
                    x2.extend(bbx[...,2].tolist())
                    y2.extend(bbx[...,3].tolist())

                    ids.extend(bbx[...,4].astype('int').tolist())
                    confs.extend(bbx[...,5].tolist())
                    clss.extend(bbx[...,6].astype('int').tolist())


                    Output_message.x_min=x1
                    Output_message.y_min=y1
                    Output_message.x_max=x2
                    Output_message.y_max=y2
                    Output_message.classes=clss
                    Output_message.track_ID=ids
                    Output_message.confidence=confs
                    return Output_message


def annotate(img,bbx,color = (0, 0, 255)):   
                    img=np.ascontiguousarray(img,np.uint8)             
                    xyxys = bbx[:, 0:4].astype('int') # float64 to int
                    ids = bbx[:, 4].astype('int') # float64 to int
                    confs = bbx[:, 5]
                    clss = bbx[:, 6].astype('int') # float64 to int
                    inds = bbx[:, 7].astype('int') # float64 to int
                    thickness = 2
                    fontscale = 2

                    for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                        
                        cv2.rectangle(
                            img,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color,
                            thickness
                        )
                        cv2.putText(
                            img,
                            f'id: {id}, conf: {round(conf,2)}, class: {cls}',
                            (xyxy[0], xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale,
                            color,
                            thickness
                        )
                    return img
def track(bbx,tracker,img):        
            tracks_bbx = tracker.update(bbx, img)
            if tracks_bbx.shape[0]>0:
             if tracks_bbx.shape[1]<7:pass     
             else:
                tracked_bbx=tracks_bbx.copy()
                return tracked_bbx
             
def filter_posts(bboxes, img_width=2400, img_height=1080):
    # Calculate centroids of the bounding boxes
    centroids_x = (bboxes[:, 0] + bboxes[:, 2]) / 2  # Calculate x-coordinate of centroids
    centroids_y = (bboxes[:, 1] + bboxes[:, 3]) / 2  # Calculate y-coordinate of centroids

    # Thresholds for the upper-right region
    upper_right_threshold_x = 0.5 * img_width  # Adjust as needed
    upper_right_threshold_y = 0.3 * img_height  # Adjust as needed

    # Filter centroids based on upper-right region conditions
    filter_condition = (centroids_x > upper_right_threshold_x) & (centroids_y < upper_right_threshold_y)
    filtered_bboxes = bboxes[~filter_condition]  # Keep boxes that do not meet the condition

    return filtered_bboxes

