from flask import Flask, render_template, Response
import cv2
import FaceDetector as detector
from server import imageObject
import Augmentation as ag

app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(imageObject("alok1.png"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)