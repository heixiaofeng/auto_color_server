from flask import Flask
from flask_restful import Api

import modules.globals as g
from colorize.main import Colorize

app = Flask(__name__)
api = Api(app, prefix='/api/v1')
app.config['UPLOAD_FOLDER'] = g.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = g.MAX_FILE_SIZE_MB * 1024 * 1024

api.add_resource(Colorize, '/colorize')

if __name__ == '__main__':
    app.run(host='10.0.0.202')
