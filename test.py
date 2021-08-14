from flask import Flask, request, send_from_directory, json, Response, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import time

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = 'images/'
app.config['DOWNLOAD_FOLDER'] = 'videos/'


def random_name():
    return uuid.uuid4().hex

def send_result(response=None, error='', status=200):
    if response is None: response = {}
    result = json.dumps({'result': response, 'error': error})
    return Response(status=status, mimetype="application/json", response=result)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/videos/<path:filename>', methods=['GET','POST'])
def download(filename):
    downloads = app.config['DOWNLOAD_FOLDER']
    return send_from_directory(directory=downloads, filename=filename)


@app.route('/', methods=['GET', 'POST'])
def root():
	return "Face Swap is running..."

# Health checkup required by GCP App Engine
@app.route('/liveness_check', methods=['GET', 'POST'])
def liveness_check():
	return '', 200


@app.route('/readiness_check', methods=['GET', 'POST'])
def readiness_check():
	return '', 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    start = time.time()
    # Input selection
    # check if the post request has the file part
    file = None
    try:
        file = request.files['image']
    except KeyError:
        return send_result(error='Please make sure to send image file with key `image`!', status=422)
    if file.filename == '':
        return send_result(error='No selected file', status=422)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # input_video = 'input.mp4'
        # # Output file names
        # # The first generated result with no audio will be called:
        # out_no_audio = 'no_audio.mp4'
        # # After combining audio will be called:
        # out_normal = 'output_gcp.mp4'
        # # Second variant with lip synced alogrithm be called:
        # out_lip_synced = 'output_lip.mp4'
        # # Audio extraction file will be called:
        # audio = 'input.mp3'
        # #Resize image and video to 256x256
        # cropped_image = preprocess_image(source_image)
        # # Extract audio
        # video2mp3(input_video, audio)
        # source_video = preprocess_video(input_video)
        # # video saved to file after complete
        # make_video_prediction(cropped_image, source_video, out_no_audio)
        # video_add_mp3(out_no_audio, audio, out_normal)

        video = 'output_gcp.mp4'
        time.sleep(10)
        # send_file(source_image, as_attachment=True)
        processing = '{:.2f}'.format((time.time() - start) / 1000)
        return send_result(response={
            'id': random_name(),
            'processing_time': str(processing) + 's'
        }, status=200)
    else:
        return send_result(error='Please make sure you send only image file with key `image`!', status=422)


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=False)