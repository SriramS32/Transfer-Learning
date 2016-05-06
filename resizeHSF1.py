import os, sys
from PIL import Image

size = 28, 28
pathToImageFolder = "/Users/sriramsomasundaram/Desktop/cluster/EmotionDataset/cohn-kanade-images/S005/001/"
pathToDestinationFolder = "/Users/sriramsomasundaram/Desktop/MergedResized/"

files_in_dir = os.listdir(pathToImageFolder)
count = 0
for file_in_dir in files_in_dir:
    if count != 0:

    #outfile = os.path.splitext(infile)[0] + "out.bmp"
    #outfile2 = os.path.splitext(infile)[0] + "out2.bmp"
    	outfile = pathToDestinationFolder + os.path.splitext(file_in_dir)[0] + ".bmp"
    #print outfile
        im = Image.open(pathToImageFolder+file_in_dir)
        #im.thumbnail(size, Image.ANTIALIAS)
        #im.save(outfile, 'BMP')

        #half_the_width = im.size[0] / 2
        #half_the_height = im.size[1] / 2
        #img2 = im.crop((half_the_width - 50,half_the_height - 50,half_the_width + 50,half_the_height + 50))
        #img2.save(outfile2)

        im = im.resize(size, Image.ANTIALIAS)
        im.save(outfile,'BMP')

        #img3 = im.crop((0,0,100,100))
        #img3.save(outfile3,'BMP')
    count+=1
