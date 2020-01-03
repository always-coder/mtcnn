import cv2
from src.detector import detect_faces
from src.utils import show_bboxes
from PIL import Image
import time
import numpy

def get_image_from_camera():
    capture = cv2.VideoCapture(0)

    while 1:
        ret, frame = capture.read()
        window_name = "face"
        #cv2.imshow(window_name, frame)
        '''
        bounding_boxes, landmarks = detect_faces(frame)
        image = show_bboxes(image, bounding_boxes, landmarks)
        '''
        
        if (ret):
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv_img)
            image = image.resize((200, 300),Image.ANTIALIAS) 
            bounding_boxes, landmarks = detect_faces(image)
            
            image = show_bboxes(image, bounding_boxes, landmarks)
            
            img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
            cv2.imshow("face detect", img) 
        if cv2.waitKey(100) & 0xff == ord('q'):
            break;

    capture.release()
    cv2.destroyAllWindows()
        
def main():
    get_image_from_camera()



if __name__ == "__main__":
    main()
