from flask import Flask, render_template, request, url_for
from analysis.image import capture_image, analyze_image
from analysis.communication import send_sms
from analysis.speech import SpeechListener
from dask import delayed, compute
from jinja2 import Template

speech_score = 0
name = ""

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'GET' or request.method == 'POST':
        return render_template("contactForm.html")


@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    global name
    if request.method == "POST":
        name = request.form['fname']
        return render_template("webcam.html")


@app.route('/speech', methods=['GET', 'POST'])
def speech_to_text():
    if request.method == "POST":

        @delayed
        def image_capture_wrapper():
            capture_image()

        @delayed
        def speech_wrapper():
            listener = SpeechListener()
            human = listener.listen()
            print('DONE')
            return human

        scores = [image_capture_wrapper(), speech_wrapper()]
        scores = compute(scores)

        print(scores)

        return render_template("webcam.html")


@app.route('/submit', methods=['GET', 'POST'])
def analysis():
    global speech_score
    if request.method == "POST":
        if speech_score <= 50:
            return render_template("QuestionnaireBad.html")
        else:
            return render_template("QuestionnaireGood.html")


@app.route('/summary', methods=['GET', 'POST'])
def summary():
    global name
    if request.method == "POST":
        emotions_summary = analyze_image("C:/Users/MAIN/PycharmProjects/Self.ai/tmp/imgopencv1.png")
        happiness = emotions_summary[0]
        calm = emotions_summary[1]
        sadness = emotions_summary[2]
        anger = emotions_summary[3]
        fear = emotions_summary[4]
        send_sms(speech_score, happiness, calm, sadness, anger, fear, name)
        return render_template("summary.html", score=speech_score, lst=emotions_summary)


if __name__ == '__main__':
    app.run()
