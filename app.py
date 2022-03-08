from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    img = image.load_img(img_path , target_size = (512,512))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    images = np.vstack([img])
    p = model.predict(images)
    classes = np.argmax(p, axis = 1)
    a = 'Acceptable IQ'
    b = 'Unacceptable IQ'
    if p == 0:
        return a
    else:
        return b

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Good luck predicting !!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)