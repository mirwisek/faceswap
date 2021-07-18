import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from skimage import img_as_ubyte
import subprocess	# used for audio extraction only

from flask import Flask

app = Flask(__name__)

import sys
# Change folder so the dependencies can find files, easily
sys.path.insert(0, 'first_order')

from demo import load_checkpoints
from demo import make_animation

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
	out_lip_synced = 'output_lip.mp4'
	audio = 'input.mp3'
	#Resize image and video to 256x256
	cropped_image = preprocess_image(source_image)
	# Extract audio
	video2mp3(input_video, audio)
	source_video = preprocess_video(input_video)
	# video saved to file after complete
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

@app.route('/', methods=['GET', 'POST'])
def root():
	return "Face Swap is running..."


# @app.route('/predict', methods=['GET', 'POST'])
# def root():
# 	return "Face Swap is running..."


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=False)