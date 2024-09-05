import os
import cv2
import numpy as np
from face_recognition import face_locations, face_encodings
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
import logging
import yaml
from werkzeug.utils import secure_filename
from PIL import Image
import io
import time
import redis
from functools import lru_cache

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Determine the environment
ENV = os.environ.get('FLASK_ENV', 'development')

if ENV == 'production':
    redis_client = redis.Redis(host='redis', port=6379, db=0)
    def cache_get(key):
        return redis_client.get(key)
    def cache_set(key, value):
        redis_client.set(key, value)
else:
    in_memory_cache = {}
    def cache_get(key):
        return in_memory_cache.get(key)
    def cache_set(key, value):
        in_memory_cache[key] = value

# Modify the lru_cache decorator to use Redis in production
def redis_cache(func):
    def wrapper(*args, **kwargs):
        if ENV == 'production':
            key = str(args) + str(kwargs)
            result = redis_client.get(key)
            if result is not None:
                return np.frombuffer(result, dtype=np.float64)
            result = func(*args, **kwargs)
            redis_client.set(key, result.tobytes())
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

# Set up logging
logging.basicConfig(level=logging.DEBUG if ENV == 'development' else logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# LRU cache for face encodings
@lru_cache(maxsize=1000)
@redis_cache
def get_face_encoding(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_encodings(rgb_image)
    return encodings[0] if encodings else None

def process_image(image_path):
    try:
        start_time = time.time()

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Convert to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_locations(rgb_image, model=config['face_detection']['model'], number_of_times_to_upsample=config['face_detection']['upsample'])

        # Get face encodings
        encodings = [get_face_encoding(image_path) for _ in faces]

        processing_time = time.time() - start_time
        logger.info(f"Processed image {image_path} in {processing_time:.2f} seconds")

        return len(faces), encodings
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config['allowed_extensions']

@app.route('/count_faces', methods=['POST'])
def count_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join(config['temp_dir'], filename)

        # Save the image
        try:
            image.save(image_path)
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return jsonify({'error': 'Error saving image'}), 500

        # Process the image
        face_count, encodings = process_image(image_path)

        # Clean up
        os.remove(image_path)

        if face_count is not None:
            return jsonify({
                'face_count': face_count,
                'encodings': [enc.tolist() if enc is not None else None for enc in encodings]
            })
        else:
            return jsonify({'error': 'Error processing image'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs(config['temp_dir'], exist_ok=True)
    # Initialize ThreadPoolExecutor 
    executor = ThreadPoolExecutor(max_workers=config['max_workers'])
    # Run the app with debug mode in development
    app.run(debug=ENV == 'development', host=config['host'], port=config['port'])