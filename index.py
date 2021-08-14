import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import subprocess	# used for audio extraction only
from flask import Flask, send_from_directory, request, json, Response, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import time

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'images/'
DOWNLOAD_FOLDER = 'videos/'

import sys
# Change folder so the dependencies can find files, easily
sys.path.insert(0, 'first_order')

from demo import load_checkpoints
from demo import make_animation

# Make sure folders exist
def make_dir(directory):
	if not os.path.exists(directory): os.makedirs(directory)

make_dir(UPLOAD_FOLDER)
make_dir(DOWNLOAD_FOLDER)

sys.path.insert(0, 'wav2lip')

# import wav2lip.inference as lipsync

face_weights_dir = 'face_weights/'
config_dir = 'first_order/config/'
fps = None

config = {
	'bair' : ('bair-256.yaml','bair-cpk.pth.tar'),
	'vox' : ('vox-256.yaml', 'vox-cpk.pth.tar'),
	'vox-adv' : ('vox-adv-256.yaml', 'vox-adv-cpk.pth.tar')
}

# Change the key to reflect the configuration
using_config = config['vox']

# Generate random video name
def random_name():
    return uuid.uuid4().hex

# Check if input image is an allowed extension file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Utility function to send responses to client
def send_result(response=None, error='', status=200):
    if response is None: response = {}
    result = json.dumps({'result': response, 'error': error})
    return Response(status=status, mimetype="application/json", response=result)


# Download videos
@app.route('/videos/<path:filename>', methods=['GET','POST'])
def download(filename):
    downloads = DOWNLOAD_FOLDER
    return send_from_directory(directory=downloads, filename=filename + '.mp4')

# Load face model weights
generator, kp_detector = load_checkpoints(
	config_path = config_dir + using_config[0], 
	checkpoint_path = face_weights_dir + using_config[1]
)

def center_crop(img):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
  
    shape = img.shape
    dim = (shape[0], shape[1])
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img
    

def preprocess_frame(frame, shape=(256,256)):
    res = center_crop(frame)
    return resize(res, shape)

# Open and preprocess image
def preprocess_image(input_image):
	source_image = imageio.imread(input_image)
	cropped_image = preprocess_frame(source_image, (256, 256))[..., :3]
	return cropped_image

# Open and preprocess video
def preprocess_video(input_video):
	global fps
	reader = imageio.get_reader(input_video)
	fps = reader.get_meta_data()['fps']
	driving_video = []
	try:
		for im in reader:
			driving_video.append(im)
	except RuntimeError:
		pass
	reader.close()
	# List comprehension
	driving_video = [preprocess_frame(frame, (256, 256))[..., :3] for frame in driving_video]
	return driving_video


def make_video_prediction(source_image, driving_video, output_vid_file):
	predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
	#save resulting video
	imageio.mimsave(output_vid_file, [img_as_ubyte(frame) for frame in predictions], fps=fps)
	print('Completed')


def video2mp3(file_name, outfile_name):
	"""
		Convert video to audio
		:param file_name: the path of the incoming video file
		:param outfile_name: specify output file otherwise will be extracted as input_name.mp3
		:return:
	""" 
	if outfile_name is None: 
		outfile_name = file_name.split('.')[0] + '.mp3'
	subprocess.call('ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name, shell=True)


def video_add_mp3(source_vid, mp3_file, output_vid):
	"""
		Video add audio
		:param source_vid: The path of the incoming video file
		:param mp3_file: the path of the incoming audio file
		:return:
	"""
	subprocess.call('ffmpeg -i ' + source_vid
                    + ' -i ' + mp3_file + ' -strict -2 -f mp4 '
                    + output_vid, shell=True)

# =====		END OF AUDIO SECTION	=====

def main():
	source_image = 'input.jpg'
	input_video = 'input.mp4'
	out_no_audio = 'no_audio.mp4'
	out_normal = 'output_gcp.mp4'
	# out_lip_synced = 'output_lip.mp4'
	audio = 'input.mp3'
	# #Resize image and video to 256x256
	cropped_image = preprocess_image(source_image)
	# # Extract audio
	video2mp3(input_video, audio)
	source_video = preprocess_video(input_video)
	# # video saved to file after complete
	make_video_prediction(cropped_image, source_video, out_no_audio)
	video_add_mp3(out_no_audio, audio, out_normal)

	# Produce lipsync result
	# args = lipsync.Arguments(
	# 	load_checkpoints='wav2lip_weights/wav2lip.pth',
	# 	face=out_no_audio,
	# 	audio=audio,
	# 	outfile=out_lip_synced
	# )
	# lipsync.predict(args)
	pass

@app.route('/', methods=['GET', 'POST'])
def root():
	return "Face App is running..."


# Health checkup required by GCP App Engine
@app.route('/liveness_check', methods=['GET', 'POST'])
def liveness_check():
	return '', 200


@app.route('/readiness_check', methods=['GET', 'POST'])
def readiness_check():
	return '', 200


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    start = time.now()
    # Input selection
    # check if the post request has the file part
    source_image = None
    try:
        source_image = request.files['image']
    except KeyError:
        return send_result(error='Please make sure to send image file with key `image`!', status=422)
    if source_image.filename == '':
        return send_result(error='No selected file', status=422)

    if source_image and allowed_file(source_image.filename):
        filename = secure_filename(source_image.filename)
        source_image.save(os.path.join(UPLOAD_FOLDER, filename))
        input_video = 'input.mp4'
        # Output file names
        # The first generated result with no audio will be called:
        out_no_audio = 'no_audio.mp4'
        # After combining audio will be called:
        ouptut_id = random_name()
        out_normal = '{}.mp4'.format(ouptut_id)
        # Second variant with lip synced alogrithm be called:
        out_lip_synced = 'output_lip.mp4'
        # Audio extraction file will be called:
        audio = 'input.mp3'
        #Resize image and video to 256x256
        cropped_image = preprocess_image(source_image)
        # Extract audio
        video2mp3(input_video, audio)
        source_video = preprocess_video(input_video)
        # video saved to file after complete
        make_video_prediction(cropped_image, source_video, out_no_audio)
        video_add_mp3(out_no_audio, audio, out_normal)

        processing = '{:.2f}s'.format((time.time() - start) / 1000)
        return send_result(response={
            'id': ouptut_id,
            'processing_time': processing
        }, status=200)
    else:
        return send_result(error='Please make sure you send only image file with key `image`!', status=422)


if __name__ == '__main__':
	# main()
	app.run(host='0.0.0.0', port=8080, debug=False)