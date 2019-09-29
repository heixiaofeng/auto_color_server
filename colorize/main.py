import uuid
from os import path

from flask import request, current_app, send_from_directory
from flask_restful import Resource, abort
from werkzeug.datastructures import FileStorage

from colorize.painter import Painter

# FIXME PNG only
ALLOWED_EXTENSIONS = set(['png'])


class Colorize(Resource):
    def post(self):
        uuid_name = str(uuid.uuid4())
        save_file('line', f'{uuid_name}')
        save_file('ref', f'{uuid_name}')

        painter = Painter()
        painter.colorize(uuid_name)

        color_dir = path.join(current_app.root_path, 'images/color/')
        print(f'sending {color_dir}/{uuid_name}')
        return send_from_directory(directory=color_dir, filename=f'{uuid_name}.jpg')


def save_file(form_key, unique_name):
    file: FileStorage = request.files[form_key]
    if not file:
        abort(500, msg=f'Invalid parameter: {form_key}: {file}')

    ext = file.filename.rsplit('.', 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        abort(500, msg=f'Unsupported content-type: {form_key}: {file}')

    file.save(f'images/{form_key}/{unique_name}.{ext}')
