from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

def cluster_colors(image, num_clusters=5):
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria and apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Map the labels to colors
    segmented_image = centers[labels]

    # Reshape the segmented image back to the original dimensions
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, labels

def calculate_cluster_areas(labels, num_clusters):
    areas = []
    for cluster in range(num_clusters):
        area = np.sum(labels == cluster)
        areas.append(area)
    return areas

def calculate_cdr(cluster_areas):
    # Assuming cluster 0 corresponds to optic disc and cluster 1 corresponds to optic cup
    disc_area = cluster_areas[1]
    cup_area = cluster_areas[0]
    cdr = cup_area / disc_area
    return cdr

def calculate_risk(cdr):
    return "High risk" if cdr > 0.5 else "Low risk"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            num_clusters = 4
            segmented_image, labels = cluster_colors(image_rgb, num_clusters)
            cluster_areas = calculate_cluster_areas(labels, num_clusters)
            cdr = calculate_cdr(cluster_areas)
            risk = calculate_risk(cdr)
            return render_template('index.html', message='Risk of glaucoma: {}'.format(risk))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
