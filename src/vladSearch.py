import glob
import cv2 
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import time
import faiss 


#Computer vision and machine learning approach with SIFT,K-Means, VLAD, PCA, Faiss
class VladSearch():
    
    def __init__(self, root, imagesPath, descriptorPCA = False, vladPCA=False, numDescriptors = 500):
        self.root = root
        self.imagesPath = imagesPath

        self.imageUris = [f for f in sorted(glob.glob(self.imagesPath + '/*'))]
        self.numDescriptors = numDescriptors
        self.descriptorPCA = descriptorPCA
        self.vladPCA = vladPCA
        
        #self.imagesPath = '/Users/ryanbrandt/Documents/WebScrape/BookPictures/'
   
        self.descriptorsPath = os.path.join(root,"Descriptors")
        self.vladsPath = os.path.join(root,"Vlads")
        self.modelsPath = os.path.join(root,"Models")
        self.kmeansPath = os.path.join(self.modelsPath, "kmeans.pkl")
        self.objPath = os.path.join(self.modelsPath, "VladSearchObj.pkl")

        #if paths dont exist, create them 
        self.createEnv()

        self.pcaPath = os.path.join(self.modelsPath, "pca.pkl")
        self.indexPath = os.path.join(self.modelsPath, "index")
        
 	#create environment
    def createEnv(self):
    	if not os.path.isdir(self.descriptorsPath):
    		os.makedirs(self.descriptorsPath)

    	if not os.path.isdir(self.vladsPath):
    		os.makedirs(self.vladsPath)

    	if not os.path.isdir(self.modelsPath):
    		os.makedirs(self.modelsPath)

    #clean the images folder, delete is defaulted to true
    def cleanImages(self, delete=True):
        #check if the image can be opened

        uris = [f for f in sorted(glob.glob(self.imagesPath + '/*'))]
        extractor = cv2.xfeatures2d.SIFT_create(self.numDescriptors)
        for uri in uris:
            try: 
                kps, des = self.extract(uri, extractor, resize=True, rootsift=True)
                s = des.size
            except:
                print(uri + " can't be extracted from")
                if delete == True: 
                    os.remove(uri) 
                    print(uri + " was deleted")


    def loadAllDescriptorsAndKeypoints(self,keypoints):
        
        kps = [
                cv2.KeyPoint(x=t[0][0], y=t[0][1], _size=t[1], _angle=t[2],
                             _response=t[3], _octave=t[4], _class_id=t[5])
                for k in kps
            ]
    #Utility: save single file
    def save(self, item, uri):
        with open(uri, 'wb') as pkl:
            pickle.dump(item, pkl)
    
    #Utility: load single file
     #openCV keypoints cannot be pickled, so we have to do this work around 
    def load(self, uri, keypoints= False):
        with open(uri, 'rb') as pkl:
            item = pickle.load(pkl)
            if keypoints: 
                temp, des = item
                kps = [
                    cv2.KeyPoint(x=k[0][0], y=k[0][1], _size=k[1], _angle=k[2],
                                 _response=k[3], _octave=k[4], _class_id=k[5])
                    for k in temp
                ]
                item = (kps,des)
        return item
    
    #Utility: loads all items in a folder
    #openCV keypoints cannot be pickled, so we have to do this work around 
    def loadAll(self,folder, keypoints= False):
        uris = [f for f in sorted(glob.glob(folder + '/*'))]
        items = [self.load(uri) for uri in uris]
        if keypoints: 
            temps, des = zip(*items)
            kps = [
                cv2.KeyPoint(x=k[0][0], y=k[0][1], _size=k[1], _angle=k[2],
                             _response=k[3], _octave=k[4], _class_id=k[5])
                
                for t in temps
                for k in t
            ]
            items = (kps,des)
        return items, uris
    
    #resize gray scale image to a width of 320 and proportional height with respect to the width
    def resizeImage(self, image):
        image_height, image_width = image.shape
        resizeWidth = 320
        resizeHeight = int(image_height * resizeWidth / image_width)
        resizedImage = cv2.resize(image, (resizeWidth, resizeHeight) , interpolation=cv2.INTER_AREA)
        return resizedImage

    def rootsift(self, des, eps=1e-7):
        if des is not None:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)
        return des
    
    #crops image (removing noise)
    def crop(self, gray):
        
        image_height, image_width = gray.shape

        y= int(image_height * .15)
        x= int(image_width * .05)
        h= int(image_height * .75)
        w= int(image_width * .9)
        crop = gray[y:y+h, x:x+w]

        plt.imshow(crop,),plt.show()
        return crop

    #extract sift features, rootsift is on 
    def extract(self, uri, extractor, resize=True, rootsift=True, crop=False):
        gray = cv2.imread(uri, cv2.IMREAD_GRAYSCALE)
        if resize == True:
            gray = self.resizeImage(gray)
        if crop == True:
            gray = self.crop(gray)
            
        kps, des = extractor.detectAndCompute(gray, None)
        if rootsift:
            des = self.rootsift(des)
        
        return kps, des
        
    #extract all keypoints and descriptors from files located in uris
    #note that for extremely large data sets that can not fit into main memory a patial fit fo pca is required
    def extractAll(self, pca = False, resize = True, rootsift=True):
        
        uris = [f for f in sorted(glob.glob(self.imagesPath + '/*'))] 

        print("Extracting sift features on ", len(uris), " images...")
        extractor = cv2.xfeatures2d.SIFT_create(self.numDescriptors)
        for i,uri in enumerate(uris):
            kps, des = self.extract(uri,extractor, resize=resize, rootsift=rootsift)
            u = os.path.basename(uri)[:-4] + ".pkl"
            #u = str(i)
            kps = [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in kps
            ]
            self.save((kps,des), os.path.join(self.descriptorsPath, u))
        if pca == True:
            self.applyPcaOnAll(self.descriptorsPath)
         
        print("Done. All ", len(uris), " feature arrays saved")
            
    #apply dementionality reduction on sift features from 128 to 64 dementions to save space
    #overwrite original sift descriptors and store the new length 64 representation 
    #might reqire batch processing for large data
    #whitening on!
    def applyPcaOnAll(self, folder, n_components = 64, keypoints=True):
        print("Computing PCA on all uris in folder: ", folder)
        items, uris = self.loadAll(folder, keypoints=keypoints)
        if keypoints:
              _ ,  X = items
             
        else:
            X = items
        #print(allArrays[0].reshape(1,-1).shape)
        print(np.asarray(X[0]).shape)
        pca = PCA(n_components=n_components)
        pca.fit(np.vstack(X))
        allPca = [pca.transform(a.reshape(1,-1)) for a in X]
   
        print(allPca[0].shape)
        print("Complete")
        
        for i, a in enumerate(allPca):
            self.save(a.flatten(), uris[i])
        
        self.save(pca, self.pcaPath)
        
        print("Dimentions reduced from ", X[0].shape, "to ", allPca[0].shape)
        
    #apply k-means to all descriptors, default = 128 clusters   
    def kmeans(self, n_clusters = 128):
        print("Computing Kmeans...")
        items , uris = self.loadAll(self.descriptorsPath, keypoints = True)
        allKeypoints, allDescriptors = items[0], items[1] 

        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                batch_size=1000,
                random_state=0,
                init_size=n_clusters * 3).fit(np.vstack(allDescriptors))
        
        self.save(kmeans, self.kmeansPath)
        print("Done. Saved Kmeans")
      
    #compute the vlad representations for each image
    def computeVlads(self):
        print("Computing a VLAD for each image...")
        
        items, uris = self.loadAll(self.descriptorsPath, keypoints=True)
        allKeypoints, allDescriptors = items[0], items[1]
        kmeans = self.load(self.kmeansPath) 
        #labels = [kmeans.predict(des) for des in allDescriptors]
       
        for i,des in enumerate(allDescriptors):
            v = self.computeVlad(des,kmeans)
            self.save(v, os.path.join(self.vladsPath, os.path.basename(uris[i])))
    
        print("Done. Saved all ", len(allDescriptors)," Vlads with shape: ", v.shape)
        
    #X : input all descriptors of an image 
    def computeVlad(self, X, kmeans):
        predictedLabels = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        k = kmeans.n_clusters

        m,d = X.shape
        V = np.zeros([k,d])
        #computing the differences

        
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels==i)>0:
                # add 
                V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)

        V = V.flatten()
        V = np.sign(V)*np.sqrt(np.abs(V)) # power normalization, also called square-rooting normalization
        V = V/np.sqrt(np.dot(V,V))    # L2 normalize
        
        #V = V / np.linalg.norm(V)
  
        #return np.ascontiguousarray(data.astype('float32'))
        
        return V

    def createIndex(self):
        print("Creating index...")
        
        allVlads, uris = self.loadAll(self.vladsPath)
        
        allVlads = np.asarray(allVlads).astype('float32')
        
        #allVlads = np.vstack(allVlads)
        print(allVlads[0].shape[0])
        d = allVlads[0].shape[0]
        '''
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(allVlads)
        index.add(allVlads)
        '''
        
        index = faiss.IndexFlatL2(d)   # build the index
        index.add(allVlads)    #add to the index
        
        faiss.write_index(index, self.indexPath) #save the index
        
        print("Done. Saved ",index.ntotal," vlads to the index")
   
    
    #create all Vlads and store them in faiss database
    def all(self):
        #self.extractAll()
        #self.kmeans()
        self.computeVlads()
        if self.vladPCA == True: 
            self.applyPcaOnAll(self.vladsPath, n_components = 512,keypoints=False)
        self.createIndex()

    #knn matching using OpenCv flann
    def knnMatch(self, des_q, des_t):
        ratio = 0.7  # According to Lowe's test
        flann = cv2.FlannBasedMatcher()
        # For each descriptor in des_q, find the best two matches in des_t
        two_nn = flann.knnMatch(des_q, des_t, k=2)
        # Find all the best matches that are significantly better than 
        #the second match, and get the corresponding index pair
        matches = [(first.queryIdx, first.trainIdx) for first, second in two_nn
                if first.distance < ratio * second.distance]
        return matches

    def filter(self, pt_qt):
        if len(pt_qt) > 0:
            pt_q, pt_t = zip(*pt_qt)
            # Get the transformation matrix and mask of normal points that match the coordinates
            M, mask = cv2.findHomography(np.float32(pt_q).reshape(-1, 1, 2),
                                         np.float32(pt_t).reshape(-1, 1, 2),
                                         cv2.RANSAC, 3)
            return mask.ravel().tolist()
        else:
            return []

    def spacialVerification(self, resultUris,queryDes, queryKps):
   
        keypoints , descriptors = [], []
        for uri in resultUris: 

            k, d =  self.load(os.path.join(self.descriptorsPath, os.path.basename(uri).split('.')[0]+'.pkl'), keypoints=True)
            #k, d = item[0], item[1]
            keypoints.append(k)
            descriptors.append(d)

        scores = np.zeros(len(descriptors))
        #print(scores)
        # Get matching coordinates using kNN algorithm
        pairs = [
            [(queryKps[q].pt, keypoints[i][t].pt)
             for q, t in self.knnMatch(queryDes, descriptors[i])]
            for i in range(len(descriptors))
        ]
        
        
        for i in range(len(descriptors)):
            mask = self.filter(pairs[i])
            scores[i] += np.sum(mask)
        

        top = np.argwhere(scores>20.0)
        #if np.amax(scores) > 20.0: 
        #    print(scores)
         #   print(np.amax(scores))
            #return np.argmax(scores)
        top = list(top.flatten())

        if len(top) > 0:
            print(scores)
            return list(top)
        else: 
            return []
 
     


    def displayMatches(self,uris, q_uri):
        plotAmount = len(uris) + 1 
        fig, axs = plt.subplots(1,plotAmount,figsize=(30, 30))
        image = cv2.imread(q_uri,1)
        axs[0].imshow(image)
        axs[0].set_title('Query')
        print(plotAmount)
        for i in range(1, plotAmount, 1):
            print(i)
            image = cv2.imread(uris[i -1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            axs[i].imshow(image)


        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        plt.tight_layout()
        plt.show()

    def query(self, imagePath, topKResults = 30, display= False):

        kmeans = self.load(self.kmeansPath)
   
        index = faiss.read_index(self.indexPath)
        extractor = cv2.xfeatures2d.SIFT_create(self.numDescriptors)
        kps, des = self.extract(imagePath,extractor, crop=False)
        
   
            
        v = self.computeVlad(des,kmeans)
 
        v = v.astype('float32').reshape(1,-1)

        if self.vladPCA == True: 
            pca = self.load(self.pcaPath)
            v = pca.transform(v).astype('float32')
        
        
        d, indexes = index.search(v, topKResults) 
        #print(d, indexes)

        resultUris = [self.imageUris[i] for i in indexes[0]]
        
        
        topIndexes = self.spacialVerification(resultUris, des ,kps)
        
        if len(topIndexes) > 0: 
            resultUris = [resultUris[i] for i in topIndexes]
            if display: 
                self.displayMatches(resultUris, imagePath)

            return resultUris

        else:
            return []
        
        
        #return resultUris

     

    

def saveObj(item, uri):
    with open(uri, 'wb') as pkl:
        pickle.dump(item, pkl)


root = '/Users/ryanbrandt/Documents/VladVisualSearch'
imagesPath = '/Users/ryanbrandt/Documents/VladVisualSearch/ImagesMany'
#vs = VladSearch(root, imagesPath, vladPCA=False)
#vs.cleanImages(delete=True)
#vs.all()
#saveObj(vs,vs.objPath)

queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/campr2.jpg'
queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/absjava.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/chemccll.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/anatomyLab.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/bioh1.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/brsr.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/chemspp.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/psci.jpg'
#\queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/worldofart.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/lifessdd.jpg'
#queryPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Test/campr2.jpg'
#vs.query(queryPath,display= True)


