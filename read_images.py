import cv2
import os
import matplotlib.pyplot as plt


def read_image(path1, path2):
    img1 = cv2.imread(path1 + '.tif', -1)
    '''if img1 is not None:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)'''
    img2 = cv2.imread(path2 + '.tif', -1)

    f, axarray = plt.subplots(1, 2, figsize=(20, 20))
    axarray[0].imshow(img1, cmap='gray')
    axarray[0].axis('off')
    axarray[1].imshow(img2, cmap='gray')
    axarray[1].axis('off')
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')


j = 0

path1 = '/home/amitjans/Desktop/Hourglass/data/depth/image'
path2 = '/media/amitjans/DATA/image'

while True:
    print(j)
    try:
        read_image(path1 + str(j), path2 + str(j))
    except TypeError, cv2.error:
        print("No image found")
        plt.close('all')
        j += 1
        continue
    text = raw_input("-----------------------------------------\nEnter --> Take picture \nb --> Break \nr --> "
                     "Last image\nint i --> image i\nd --> Delete\n-----------------------------------------\n")
    if text == 'b':
        print('Breaking...')
        break

    if text == "d":
        print("Deleting...")
        for ext in (".csv", ".pcd", ".png", ".tif"):
            os.remove(path + str(j) + ext)

    j += 1

    try:
        j = int(text)
        print('Changing value...')
    except ValueError:
        pass
