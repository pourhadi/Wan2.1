import os
import sys
import logging
import argparse
import tempfile
import uuid
from datetime import datetime
import requests
from io import BytesIO
from urllib.parse import urlparse

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from wan.utils.utils import cache_video

import wan
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES, SIZE_CONFIGS

# Default configuration
DEFAULT_CONFIG = {
    "task": "i2v-14B",  # Default task (text-to-video with 14B model)
    "size": "832*480",  # Default output size
    "sample_steps": 30,  # Default sampling steps
    "sample_shift": 3.0,  # Default sampling shift
    "frame_num": 81,     # Default frame number
    "sample_solver": "unipc",  # Default solver
    "sample_guide_scale": 5.0,  # Default guidance scale
    "ckpt_dir": None,     # Will be set from command line args
    "offload_model": True,  # Default offload setting
}


cfg = WAN_CONFIGS["i2v-14B"]

wan_i2v = None

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# Global model instance
model = None

def init_model(ckpt_dir):
    global wan_i2v
    wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )


def download_image(url):
    """Download an image from a URL and return as PIL Image"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def save_temp_image(image):
    """Save PIL image to a temporary file and return the path"""
    temp_dir = os.path.join(tempfile.gettempdir(), "wan_server")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename for the temporary image
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(temp_dir, filename)
    
    # Save the image
    image.save(filepath)
    return filepath

def generate_video(image_path, prompt, task="t2v-14B", size="1280*720"):
    """Generate video using the Wan model"""
    global model, DEFAULT_CONFIG, wan_i2v

    video = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[size],
            frame_num=81,
            shift=3.0,
            sample_solver="unipc",
            sampling_steps=30,
            guide_scale=5.0,
            seed=-1,
            offload_model=False)

    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
    suffix = '.png' if "t2i" in task else '.mp4'
    save_file = f"{task}_{formatted_prompt}_{formatted_time}" + suffix

    cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    return save_file

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Endpoint to generate a video from an image URL and text prompt"""
    # Get parameters from request
    request_data = request.json
    
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Get image URL and prompt from request
    image_url = request_data.get('image_url')
    prompt = request_data.get('prompt')
    

    # Validate input
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # For image-to-video tasks, download the image
        image_path = None
        image = download_image(image_url)
        image_path = save_temp_image(image)
        
        # Generate video
        output_path = generate_video(image_path, prompt, task, size)
        
        # Return the video file
        return send_file(output_path, mimetype='video/mp4')
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Error generating video: {str(e)}")
        return jsonify({"error": f"Failed to generate video: {str(e)}"}), 500

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.1 Video Generation API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Update default config with checkpoint directory
    DEFAULT_CONFIG["ckpt_dir"] = args.ckpt_dir
    
    # Initialize the Wan model
    logging.info(f"Initializing Wan model from checkpoint directory: {args.ckpt_dir}")
    
    init_model(args.ckpt_dir)

    # Start the Flask server
    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False) 