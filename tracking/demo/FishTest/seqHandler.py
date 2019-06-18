import os

directory = './img'
for fileName in os.listdir(directory):
    newName = os.path.join(directory, str(int(
        fileName.split('.')[0])).zfill(4)+'.'+fileName.split('.')[1])
    os.rename(os.path.join(directory, fileName), newName)
