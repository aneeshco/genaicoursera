# install whisper using pip install git+https://github.com/openai/whisper.git
from flask import Flask, request
import whisper
app = Flask(__name__)
model = whisper.load_model("base")
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    audio_file = request.files["audio"]
    result = model.transcribe(audio_file)
    return {"transcription": result["text"]}
if __name__ == "__main__":
    app.run(debug=True)


"""
# installing required libraries in my_env
pip install transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1
We need to install ffmpeg to be able to work with audio files in python.
sudo apt update
sudo apt install ffmpeg -y
"""