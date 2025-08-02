from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from flask import Blueprint, request, url_for
from flask import render_template
from werkzeug.utils import secure_filename
from . import app
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.switch_backend('agg')
views = Blueprint("views", __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        file = request.files['file']
        input_img = secure_filename(file.filename)
        file.save(app.config['IMAGE_UPLOADS']+input_img)

        pred = predict_save(input_img)
        return render_template('home.html', pred=pred, input_img=input_img, pred_img='pred_img.png')
    return render_template('home.html')


@views.route('/about')
def about():
    return render_template('about.html')


##############################################
model = load_model(app.config['MODEL'])
class_names = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']


def predict_save(img):
    my_image = load_img(
        app.config['IMAGE_UPLOADS']+img, target_size=(128, 128))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, 0)

    out = np.round(model.predict(my_image)[0], 2)
    fig = plt.figure(figsize=(7, 4))
    plt.barh(class_names, out, color='lightgray',
             edgecolor='red', linewidth=1, height=0.5)

    for index, value in enumerate(out):
        plt.text(value/2 + 0.1, index, f"{100*value:.2f}%", fontweight='bold')

    plt.xticks([])
    plt.yticks([0, 1, 2, 3], labels=class_names,
               fontweight='bold', fontsize=14)
    fig.savefig('pred_img.png', bbox_inches='tight')

    name = app.config['IMAGE_UPLOADS']+'pred_img.png'
    fig.savefig(name, bbox_inches='tight')

    return class_names[np.argmax(out)]
