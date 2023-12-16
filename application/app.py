from flask import Flask, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import io
import urllib
import base64
from app_script import ImagePredictor
import os

app = Flask(__name__)

model_path = '/home/prgrmcode/app/model/model.pth'
predictor = ImagePredictor(model_path)

@app.route('/')
def show_images():
    image_names = ['predict1.webp', 'predict2.webp', 'predict3.jpg', 'predict4.webp', 'predict5.jpg', 'predict6.jpg']
    images_data = []
    for image_name in image_names:
        image_path = image_name

        # Create a new matplotlib figure
        figure = Figure()
        axis = figure.subplots()

        # Load the image
        image = Image.open(image_path)
        axis.imshow(image)

        # Make a prediction
        predicted_class_name = predictor.predict_class_name(image_path)
        axis.set_title(f'Predicted class: {predicted_class_name}')

        # Convert the figure to an image
        output = io.BytesIO()
        FigureCanvas(figure).print_png(output)
        image_data = urllib.parse.quote(base64.b64encode(output.getvalue()))

        images_data.append(image_data)

    return render_template('show_images.html', images_data=images_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)