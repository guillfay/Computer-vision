import cv2
import mediapipe as mp
import time

class detectionMains():
    #Initialise et appelle le module MediaPipe
    def __init__(self,mode=False,max_mains=2,detection_accurracy=0.5,track_accurracy=0.5):
        #Valeurs laissées par défaut, deux mains détectables au maximum
        self.mode=mode
        self.max_mains=max_mains
        self.detection_accurracy=detection_accurracy
        self.track_accurracy=track_accurracy
        
        #Appelle le module MediaPipe Hands
        self.mpMains=mp.solutions.hands
        self.mains=self.mpMains.Hands()
        #Active la fonction qui relie graphiquement les Landmarks
        self.mpTrack=mp.solutions.drawing_utils

    #Appelle le flux vidéo et autorise l'affichage en superposition des Landmarks sur la main
    def trackMains(self,img,affichage=True):
        #Inversion des couleurs du flux vidéo
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #Processs de l'image acquise pour stocker la position des Landmarks
        self.resultats=self.mains.process(imgRGB)
        #Affichage des Landmarks sur la main
        if self.resultats.multi_hand_landmarks:
            for handlandmarks in self.resultats.multi_hand_landmarks:
                if affichage:
                    self.mpTrack.draw_landmarks(img,handlandmarks,self.mpMains.HAND_CONNECTIONS)
        
        #Renvoie l'image du flux vidéo ayant subi le process
        return img

    #Renvoie la position normalisée de chacun des Landmarks de chaque main
    def positionMains(self,img,nb_mains=0,affichage=True):
        # Liste des mains 1 et 2, stockant l'identifiant de chaque Landmark et ses coordonnées normalisées à la taille du flux vidéo
        lmListe1=[]
        lmListe2=[]

        if self.resultats.multi_hand_landmarks:
            #Cas ou 1 seule main visible
            if len(self.resultats.multi_hand_landmarks)==1:
                maMain=self.resultats.multi_hand_landmarks[nb_mains]
                #Parcours les Landmarks de la main considérée
                for id, lm in enumerate(maMain.landmark):
                    L,l,h=img.shape #Dimensions du cadre vidéo
                    cx,cy=int(lm.x*l),int(lm.y*L)   #Normalisation des coordonnées du Landmark
                    #print(id,cx,cy)
                    lmListe1.append([id,cx,cy])
                    #Trace un cercle jaune sur les Landmarks
                    if affichage:   
                        cv2.circle(img,(cx,cy),7,(0,255,255),cv2.FILLED)

            if len(self.resultats.multi_hand_landmarks)==2:
                maMain1=self.resultats.multi_hand_landmarks[nb_mains]
                maMain2=self.resultats.multi_hand_landmarks[nb_mains+1]
                for id, lm in enumerate(maMain1.landmark):
                    L,l,h=img.shape
                    cx,cy=int(lm.x*l),int(lm.y*L)
                    #print(id,cx,cy)
                    lmListe1.append([id,cx,cy])
                    if affichage:   
                        cv2.circle(img,(cx,cy),7,(0,255,255),cv2.FILLED)
                for id, lm in enumerate(maMain2.landmark):
                    L,l,h=img.shape
                    cx,cy=int(lm.x*l),int(lm.y*L)
                    #print(id,cx,cy)
                    lmListe2.append([id,cx,cy])
                    if affichage:   
                        cv2.circle(img,(cx,cy),7,(0,255,0),cv2.FILLED)  
        return [lmListe1,lmListe2]



    

def main():
    capture=cv2.VideoCapture(0)
    t_av=0
    t_ap=0
    detection=detectionMains()
    while True:
        success,img=capture.read()
        img=detection.trackMains(img)
        [lmListe1,lmListe2]=detection.positionMains(img)
        """if len(lmListe1)!=0:
            print(lmListe1[4])
        if len(lmListe2)!=0:
            print(lmListe2[4])"""

        # Affichage fps
        t_ap=time.time()
        fps=1/(t_ap-t_av)
        t_av=t_ap
    
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
