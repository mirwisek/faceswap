from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def root():
	return "Hello World is running..."


if __name__ == '__main__':
	# main()
    app.run(host='0.0.0.0', port=8080, debug=False)
    print('Started application')