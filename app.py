from flask import Flask, request, jsonify
from flask_redis import FlaskRedis
from werkzeug.utils import secure_filename
from flask_cors import CORS
from uuid import uuid4
from PIL import Image
import io
import os
import sys
import base64
import torch
from test import model, processor, model_ocr

# Cấu hình tải file lên server
UPLOAD_FOLDER = 'dongho'  # up file vao folder do
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
if os.path.exists(UPLOAD_FOLDER) is False:
    os.mkdir(UPLOAD_FOLDER)

# Khởi tạo app Flask
app = Flask(__name__)

# Cấu hình biến environment (env)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REDIS_URL'] = "redis://redis:6379/0"  # Điều chỉnh URL kết nối tới Redis của bạn

# Cache redis
redis_client = FlaskRedis(app)

# Cấu hình HTTP
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Hàm kiểm tra file ảnh có đuôi file .png, .jpg, .jpeg, .gif
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# API GET Hello world :v
@app.route("/")  # tao mot cai api (giao tiep giua BE va FE)
def hello_world():
    return "<p>Hello, Electronic watches!</p>"


# API [POST, GET] ảnh từ camera truyền về server
@app.route("/api/v1/upload_image", methods=['GET', 'POST'])  # tao mot cai api moi
def upload_image():
    try:
        # Xác thực Authentication token
        if 'Token' in request.headers:
            token = request.headers['Token']
            if len(token) == 36 or len(token) == 37:
                print("Pass!")
            else:
                return jsonify({'message': 'Unauthorized'}), 401
        else:
            return jsonify({'message': 'Unauthorized'}), 401

        # Request từ FE truyền về
        file = request.files['file']

        # Read the image data and encode it to base64
        img_file = file.read()
        base64_image = base64.b64encode(img_file)
        redis_client.set('mykey', base64_image)

        # Block code lưu 1000 tấm ảnh
        if len(os.listdir(path='dongho')) != 1000:  # lenh if de luu hinh anh 1000 buc anh
            if file and allowed_file(file.filename):
                filename = str(uuid4()) + '.' + secure_filename(file.filename).split('.')[1]
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return jsonify({'message': 'file is not support'}), 400

        return jsonify({'message': 'hong ok'}), 200

    except Exception as error:
        raise error


# API [POST] nhận dạng số
@app.route('/api/v1/detect', methods=['POST'])
def detect():
    try:
        # Decode the base64 image
        base64_image = redis_client.get('mykey')
        # Create a BytesIO buffer from the decoded image data
        image_data = base64.b64decode(base64_image)
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_data))

        # Kết quả nhận dạng
        results = model.predict(source=img, conf=0.2, iou=0.5)
        result = results[0]

        sodo = []
        # Tính tọa độ của bounding box trong tệp label
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            print("Object type:", class_id)
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            print("Coordinates:", cords)
            print("---")

            # Cắt ảnh con từ ảnh gốc
            cropped_image = img.crop((cords[0], cords[1], cords[2], cords[3]))

            # Convert
            pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pixel_values = pixel_values.to(device)

            # # Inference
            generated_ids = model_ocr.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            sodo.append(generated_text)

        return jsonify({'message': 'toet voi', 'base64': str(base64_image), 'sodolon': str(sodo[0]), 'sodonho': str(sodo[1])}), 200

    except Exception as error:
        raise error


# Cấu hình HTTP
@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Token')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3106)
