import flask
import urllib.request
from vladSearch import VladSearch
import pickle
from flask_cors import CORS
import os
import pandas as pd

app = flask.Flask(__name__)
CORS(app)

@app.route('/api/query', methods=['POST'])
def queryUrl():

	print('___________________________')
	print(flask.request.json)
	
	imageUrl = flask.request.json["image"]
	queryImgPath = '/Users/ryanbrandt/Documents/VladVisualSearch/ImagesQuery/querImg'

	#download image 
	urllib.request.urlretrieve(imageUrl, queryImgPath)

	#query
	vladObjPath = '/Users/ryanbrandt/Documents/VladVisualSearch/Models/VladSearchObj.pkl'
	with open(vladObjPath, 'rb') as pkl:
		vs = pickle.load(pkl)

	resultUris = vs.query(queryImgPath)

	asins = [os.path.basename(uri).split('.')[0] for uri in resultUris]

	#find image based on asin
	amzProducts = '/Users/ryanbrandt/Documents/Ebay/CSVFiles/All_Products.csv'
	df = pd.read_csv(amzProducts, index_col=None, header=0)
	images = []
	for asin in asins: 
		try: 
			img = df.loc[df['ASIN'] == asin]['Image'].values[0].split(';')[0]
			#print(img)
			images.append(img)
		except: 
			pass

	#return only top 15
	return flask.jsonify({'images': images[:15]})


#for test purposes only
@app.route('/api/test/1', methods=['GET'])
def test1():

	images=	[
    "https://images-na.ssl-images-amazon.com/images/I/51xSdEZMNgL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51xSdEZMNgL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51F2j7QbjDL.jpg",
  	]


	return flask.jsonify({'images': images})

#for test purposes only
@app.route('/api/test/2', methods=['GET'])
def test2():
	images=	[
    "https://images-na.ssl-images-amazon.com/images/I/415qr70EJ-L.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51frWLMi6bL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51EMXqR9qvL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51eV61pwbmL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/41Wh0PpcXUL.jpg",
    "https://images-na.ssl-images-amazon.com/images/I/51wibLIHR+L.jpg"
  	]


	return flask.jsonify({'images': images})

if __name__ == "__main__":
	app.run(debug=True)

