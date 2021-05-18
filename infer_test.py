import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model
model = load_model()
tokenizer = get_kobart_tokenizer()

# @app.route('/')
# def main_page():
#     return 'hello world'

@app.route('/summarization', methods=['POST'])
def summarization():
    request_data = request.get_json()
    # text = request.args('text')
    text = request_data['text']
    text = text.replace('\n', '')
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
    return jsonify({'kor_output' : output})

if __name__ == "__main__":
    app.run(debug=True)