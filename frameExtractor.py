import cv2
import os

kelas = '5'

videoDirectory = 'D:/SN/Data to Zasya/Kelas' + kelas + '.mp4'
destinationDirectory = 'D:/SN/Data to Zasya/hasilEkstrak/'+ kelas +'/'

isExist = os.path.exists(destinationDirectory)

if not isExist:
    os.makedirs(destinationDirectory)


cap = cv2.VideoCapture(videoDirectory)
i = 1
urutanFrame = 25

while (cap.isOpened()):
    _, frame = cap.read()
    if _ == False:
        break
    if ((i % urutanFrame ) == 0):
        cv2.imwrite(os.path.join(destinationDirectory, str(kelas) + '_' + str(i) + '.png'), frame)
        print('Gambar ke-' + str(i))
    i += 1
cap.release