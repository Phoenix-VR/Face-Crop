from cvzone.FaceDetectionModule import FaceDetector
import cv2
import glob

detector = FaceDetector()

source_path = 'source_imgs'
destination_path = 'cropped_faces'

# For image files 

curfiles = glob.glob(source_path + '/*.jpg')

img_no = 1

for file in curfiles:
    img = cv2.imread(file)
    img, bboxs = detector.findFaces(img,draw=False)
    
    if bboxs:
        for bbox in bboxs:
            x1 = bbox['bbox'][0]
            y1 = bbox['bbox'][1]
            x2 = bbox['bbox'][2]
            y2 = bbox['bbox'][3]
            crop_img = img[ y1:y1+y2, x1:x1+x2]
            cv2.imwrite("{}/face_img_{}.jpg".format(destination_path,img_no), crop_img)
            print('Successfully saved face from image {} out of {} images!'.format(img_no,len(curfiles)))
    else:
        print('Face not found in image {}'.format(img_no))
    img_no += 1

# For video files

curfiles = glob.glob(source_path + '/*.mp4')

img_no = 1
video_no = 1
for video in curfiles:
    cap = cv2.VideoCapture(video)

    while (cap.isOpened()):
        success, img = cap.read()
        if success == True:
            img, bboxs = detector.findFaces(img,draw=False)
            if bboxs:
                for bbox in bboxs:
                    x1 = bbox['bbox'][0]
                    y1 = bbox['bbox'][1]
                    x2 = bbox['bbox'][2]
                    y2 = bbox['bbox'][3]
                    crop_img = img[ y1:y1+y2, x1:x1+x2]
                    cv2.imwrite("{}/face_img_{}.jpg".format(destination_path,img_no), crop_img)
                    print('Successfully saved face from video {}! ({} images saved)'.format(video_no,img_no))
                    img_no += 1
            cv2.waitKey(1)
        else:
            break
    print('Video {} successfully done!'.format(video_no))
    video_no +=1
