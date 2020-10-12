import argparse

from vladSearch import VladSearch
#command line args, TODO 

'''

ap = argparse.ArgumentParser()

ap.add_argument("-query", required = True, help = "Path of a query image")

#
ap.add_argument("--top", required = True, help = "Number of images to return")


# clean the image dataset and remove the images that can't be processed 
ap.add_argument("--clean", required = True, help = "Folder path of images")

#execute the entire vlad process
ap.add_argument("--all", required = True, help = "Execute the whole Vlad process, extract descriptors, clustering (Kmeans),Vlad aggrigation, Faiss index ")

args = vars(ap.parse_args())
'''

if __name__ == '__main__':

	root = '/Users/ryanbrandt/Documents/VladVisualSearch'
	imagesPath = '/Users/ryanbrandt/Documents/VladVisualSearch/ImagesMany'
	vs = VladSearch(root, imagesPath, vladPCA=False)
	vs.cleanImages(delete=True)
	vs.all()
	saveObj(vs,vs.objPath)


	#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/campr2.jpg'
	#vs.query(queryPath,display= True)
