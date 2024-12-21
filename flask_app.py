import os
import io
import base64
from PIL import Image
from huggingface_hub import InferenceClient
#from transformers import AutoProcessor, AutoModelForCausalLM
from flask import Flask, render_template, request, redirect, url_for, send_file

#client = InferenceClient("stabilityai/stable-diffusion-2-1", token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
client = InferenceClient("stabilityai/stable-diffusion-2-1", token = "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")

output_file_path = 'output.txt'

print ('Loading VQA model')
processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

app = Flask(__name__)

name_id_to_img_num = {}

# Simulated image generation function (replace with actual logic later)
def generate_image(prompt, name_id):
	if(name_id not in name_id_to_img_num):
		name_id_to_img_num[name_id] = 0
	name_id_to_img_num[name_id] += 1

	# output is a PIL.Image object
	image = client.text_to_image(prompt)
	file_name = 'img_' + name_id + '_' + str(name_id_to_img_num[name_id]) + '.png'
	image.save(file_name)

	# Simulate image generation by returning a random image
	#return f"https://placekitten.com/200/300?random={random.randint(1, 100)}"
	#return send_file(file_name, mimetype='image/png')
	return image, file_name

# Function to handle the final submit (store or process the final data)
def final_submit(prompt, file_name):
	# Here you can save the final result or process it further
	print(f"Final Submission - Prompt: {prompt}, Image: {file_name}")
	questions = {"Does the image have any snow or ice?":1, "Does the image have a santa claus":1, "Does the image have a sleigh":1, "Does the image have a christmas tree":1, "Does the image have reindeers":1, "Does the image have christmas lights":1, "Does the image have a cartoon like style":2, "Does the image have a sketch or painting like style":2]}

	image = Image.open(file_name).convert("RGB")
	pixel_values = processor(images = image, return_tensors = "pt").pixel_values

	score = 0
	for question in questions:
		input_ids = processor(text = question, add_special_tokens = False).input_ids
		input_ids = [processor.tokenizer.cls_token_id] + input_ids
		input_ids = torch.tensor(input_ids).unsqueeze(0)
		generated_ids = model.generate(pixel_values = pixel_values, input_ids = input_ids, max_length = 50)
		answer = processor.batch_decode(generated_ids, skip_special_tokens = True)
		if(answer.lower().startswith('yes'):
			score += questions[question]
	return score

"""@app.route('/', methods=['GET', 'POST'])
def index():
	image_url = None
	prompt = ''
	name_id = ''
	if request.method == 'POST':
		name_id = request.form.get('name_id')
		prompt = request.form.get('prompt')
		if prompt:
			image_url = generate_image(prompt, name_id)
	return render_template('index.html', image_url = image_url, prompt = prompt, name_id = name_id)"""

@app.route('/', methods=['GET', 'POST'])
def index():
	data = io.BytesIO()
	encoded_img_data = base64.b64encode(data.getvalue())
	prompt = ''
	name_id = ''
	file_name = ''
	if request.method == 'POST':
		name_id = request.form.get('name_id')
		prompt = request.form.get('prompt')
		if prompt:
			image, file_name = generate_image(prompt, name_id)
			data = io.BytesIO()
			image.save(data, "png")
			encoded_img_data = base64.b64encode(data.getvalue())
	return render_template('index.html', img_data = encoded_img_data.decode('utf-8'), prompt = prompt, name_id = name_id, file_name = file_name)

@app.route('/submit', methods=['POST'])
def submit():
	prompt = request.form.get('prompt')
	file_name = request.form.get('file_name')
	score = final_submit(prompt, file_name)

	output_file = open(output_file_path, 'a')
	output_file.write(file_name + '\t' + str(score) + '\n')
	output_file.close()
	return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
	return render_template('thank_you.html')

if __name__ == "__main__":
	app.run(debug = True, host = '0.0.0.0', port = 11190)
