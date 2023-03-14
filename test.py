from flask import Flask, request, json, Response, jsonify, make_response
from apscheduler.schedulers.background import BackgroundScheduler
import random

# Instantiating the Flask application
application = Flask(__name__)

# Instantiating the scheduler for the cronjob



first_num = []
second_num = []
# Defining a cronjob function to run alongside the Flask app
# @sched.scheduled_job(trigger = 'cron', minute = 1)
def print_hello():
    global first_num, second_num 
    first_num = random.randint(1, 10)
    second_num = random.randint(1, 10)
    print(first_num+second_num)
    print('==================')


sched = BackgroundScheduler()
sched.add_job(print_hello, 'cron', minute='0-59/1')

sched.start()

print_hello()

# Defining a single API endpoint
@application.route('/test')
def test_func():
    js = json.dumps({'First': first_num, 'Second': second_num})
    return Response(json.dumps(js), status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    # Starting Flask application
    application.run(host = '0.0.0.0')