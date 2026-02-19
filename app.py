"""
Flask Application for Image Captioning
========================================
Ready to deploy to Azure

Copy this file to your local project folder as: app.py
"""

from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import io
import os

app = Flask(__name__)

# Configuration
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload

# Initialize model once on startup
print("Initializing model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
    
    # Apply float16 optimization if on GPU
    if device == 'cuda':
        model.half()
        print("Float16 optimization applied")
    
    model.eval()
    print("✓ Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Azure"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/caption', methods=['POST'])
def caption_image():
    """Generate caption for uploaded image"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'success': False
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'error': 'No image selected',
                'success': False
            }), 400
        
        # Process image
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            return jsonify({
                'error': f'Invalid image format: {str(e)}',
                'success': False
            }), 400
        
        # Generate caption
        print(f"Generating caption for image: {image_file.filename}")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Apply float16 if on GPU
        if device == 'cuda' and 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        
        with torch.inference_mode():
            out = model.generate(**inputs, max_length=50)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"Caption: {caption}")
        
        return jsonify({
            'caption': caption,
            'success': True
        }), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500

@app.route('/batch-caption', methods=['POST'])
def batch_caption():
    """Generate captions for multiple images"""
    try:
        if 'images' not in request.files:
            return jsonify({
                'error': 'No images provided',
                'success': False
            }), 400
        
        image_files = request.files.getlist('images')
        captions = []
        
        for image_file in image_files:
            try:
                image = Image.open(image_file).convert('RGB')
                
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                if device == 'cuda' and 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].half()
                
                with torch.inference_mode():
                    out = model.generate(**inputs, max_length=50)
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                captions.append({
                    'filename': image_file.filename,
                    'caption': caption
                })
            
            except Exception as e:
                captions.append({
                    'filename': image_file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'captions': captions,
            'success': True
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/info')
def info():
    """Get system information"""
    return jsonify({
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'model': 'BLIP-base',
        'optimization': 'float16' if device == 'cuda' else 'standard'
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
