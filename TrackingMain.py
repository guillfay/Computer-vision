import cv2
import mediapipe as mp
import time

capture=cv2.VideoCapture(0)

mpMains=mp.solutions.hands
mains=mpMains.Hands()
mpTrack=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success,img=capture.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resultats=mains.process(imgRGB)
    #print(resultats.multi_hand_landmarks)

    if resultats.multi_hand_landmarks:
        for handlandmarks in resultats.multi_hand_landmarks:
            for id, lm in enumerate(handlandmarks.landmark):
                L,l,h=img.shape
                cx,cy=int(lm.x*l),int(lm.y*L)
                print(id,cx,cy)
                cv2.circle(img,(cx,cy),7,(0,255,255),cv2.FILLED)
            mpTrack.draw_landmarks(img,handlandmarks,mpMains.HAND_CONNECTIONS)

    # Affichage fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

