import requests
import numpy as np
import cv2
import datetime

url = 'http://192.168.1.29:8080/shot.jpg'     #changez les chiffres avec ceux affichés sur la premiere ligne url ipv4

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)

    '''
    img_sized = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,final = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    graybis = np.array(255 * ((gray / 255) ** 2), dtype = np.uint8)
    '''

    cv2.imshow('stream', img)    # ci apres figurent les touches à presser, détail en commentaire

    if cv2.waitKey(1) == 27:   #escape, presser avant de fermer la fenetre du live
        break

    if cv2.waitKey(2) == 101:   #touche e comme enregistrer, maintenir pendant une seconde
                                # pour sauvegarder une image de ce qu'il y a à l'écran
        filename = r'C:/Mlutz/documents/images(me modifier)'   # C'est un exemple, mettez entre les guillemets
                                                  # le path de là où vous voulez enregistrer l'image.
        filename += '/capture_psi_' + str(datetime.datetime.now())[0:-7]
        cv2.imwrite(filename, img)
        print('saved')

    if cv2.waitKey(5) == 112:   #touche p comme pause, pour que l'image reste affichée afin de lancer le live
        time.sleep(10)              # le live reprend 10 sec apres avoir pressé la touche.



# le morceau suivant est à utiliser si les id des touches du clavier ne sont pas les memes sur mac que sur windows
# i.e. la sauvegarde d'image ne fonctionne pas.  (rien ne s'affiche dans le shell quand vous appuyez sur 'e', ou crash)
# changer le false qui suit en True, et le True du premier while en False pour lancer.
while False:    # Une fois lancé, appuyer sur une touche pour voir son numéro s'afficher dans la console.
    cv2.imshow('img',img)   #écrire ensuite le numéro à la place de celui qui figure
    k = cv2.waitKey(33)    # apres le 'waitkey(1) == ' dans le code pricipal.
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its value
