from flask import Flask
app = Flask(__name__)

## Rollbar init code. You'll need the following to use Rollbar with Flask.
## This requires the 'blinker' package to be installed

import os
import rollbar
import rollbar.contrib.flask
from flask import got_request_exception


with app.app_context():
    """init rollbar module"""
    rollbar.init(
        # access token
        '6cd81799ebf74af688b544466f300149',
        # environment name - any string, like 'production' or 'development'
        'flasktest',
        # server root directory, makes tracebacks prettier
        root=os.path.dirname(os.path.realpath(__file__)),
        # flask already sets up logging
        allow_logging_basic_config=False)

    # send exceptions from `app` to rollbar, using flask's signal system.
    got_request_exception.connect(rollbar.contrib.flask.report_exception, app)

## Simple flask app

@app.route('/')
def hello():
    try:
        1/0
    except Exception as e:
        rollbar.report_exc_info()
    return "Hello World!"


if __name__ == '__main__':
    app.run()