from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-detection', methods=['POST'])
def start_detection():
    input_type = request.json.get('input_type')
    video_file = request.json.get('video_file')

    if input_type == "webcam":
        subprocess.Popen(["python", "blink_detection.py", "--webcam"])
        return jsonify({"message": "Webcam detection started"}), 200
    elif input_type == "video":
        subprocess.Popen(["python", "blink_detection.py", "--video", video_file])
        return jsonify({"message": f"Video detection started with {video_file}"}), 200
    else:
        return jsonify({"message": "Invalid input"}), 400

@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    # Logic to stop the detection
    return jsonify({"message": "Detection stopped"}), 200

if __name__ == '__main__':
    app.run(debug=True)
