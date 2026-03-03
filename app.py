from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import base64

app = Flask(__name__)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    """Generate caption for a single image"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Generate caption
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_length=50)
            caption_text = processor.decode(output[0], skip_special_tokens=True)
        
        return jsonify({
            'caption': caption_text,
            'success': True
        })
    
    except Exception as e:
        print(f"Error in caption endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/batch-caption', methods=['POST'])
def batch_caption():
    """Generate captions for multiple images"""
    try:
        # Get images from request
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        captions = []
        
        # Process each image
        for file in files:
            try:
                image = Image.open(io.BytesIO(file.read())).convert('RGB')
                
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    output = model.generate(**inputs, max_length=50)
                    caption_text = processor.decode(output[0], skip_special_tokens=True)
                
                captions.append({
                    'filename': file.filename,
                    'caption': caption_text
                })
            except Exception as e:
                captions.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'captions': captions,
            'success': True
        })
    
    except Exception as e:
        print(f"Error in batch-caption endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'App is running'}), 200

@app.route('/info', methods=['GET'])
def info():
    """Get app information"""
    return jsonify({
        'app': 'Image Captioner',
        'model': 'BLIP-base',
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
