from flask import Flask, request, json, Response, send_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = 'images/'


def send_result(response=None, error='', status=200):
    if response is None: response = {}
    result = json.dumps({'result': response, 'error': error})
    return Response(status=status, mimetype="application/json", response=result)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def root():
	return "Face Swap is running..."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
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
        # send_file(source_image, as_attachment=True)
        return 'hey'
    else:
        return send_result(error='Please make sure you send only image file with key `image`!', status=422)


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=False)