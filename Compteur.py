import cv2
import mediapipe as mp
import time
import ModuleTrackingMain as mtm

#Capture du flux vidéo de la webcam
capture=cv2.VideoCapture(0)

#Initialisation des compteurs de temps avant et après
t_av=0
t_ap=0

#Appelle ModuleTrackingMain.py
detection=mtm.detectionMains()

#Définit les identifiants des Landmarks au bout de chaque doigts
boutsDoigtsID=[4,8,12,16,20]
while True:
    success,img=capture.read()
    img=detection.trackMains(img)
    [lmListe1,lmListe2]=detection.positionMains(img)
    # Stocke la liste des doigts comptabilisés pour le comptage
    doigts_tot=[]
    for lmListe in [lmListe1,lmListe2]:
        if len(lmListe)!=0:
            doigts=[]

            #Disjonction de cas entre le pouce et les autres doigts
            # Pouce
            if lmListe[boutsDoigtsID[3]][1]<lmListe[boutsDoigtsID[4]][1]:
                if lmListe[boutsDoigtsID[0]][1]<lmListe[boutsDoigtsID[0]-1][1]:
                    #On comptabilise le doigt s'il est ouvert, ie en ajoutant 1 dans la liste doigts, 0 sinon 
                    doigts.append(1)
                else:
                    doigts.append(0)
            else:
                if lmListe[boutsDoigtsID[0]][1]>lmListe[boutsDoigtsID[0]-1][1]:
                    doigts.append(1)
                else:
                    doigts.append(0)

            # Autres doigts
            for i in range(1,5):
                if lmListe[boutsDoigtsID[i]][2]<lmListe[boutsDoigtsID[i]-2][2]:
                    doigts.append(1)
                else:
                    doigts.append(0)
            #Production une liste comptant tous les doigts levés
            doigts_tot+=doigts
    #Compte tous les 1 de la liste doigts_tot, ie les doigts levés   
    totalDoigts=doigts_tot.count(1)
    
    cv2.putText(img,str(totalDoigts),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),7)



    # Affichage fps
    t_ap=time.time()
    fps=1/(t_ap-t_av)
    t_av=t_ap

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)