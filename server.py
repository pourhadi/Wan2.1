#!/usr/bin/env python3
"""
API server for image-to-video generation using Wan pipelines.
"""
import os
import argparse
import uuid
from threading import Thread
from queue import Queue
from PIL import Image
from flask import Flask, request, jsonify, url_for, send_file
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video

# Module-level parameters (set in main)
server_cfg = None
server_size = None
server_frame_num = None
server_sample_steps = None
server_sample_shift = None
server_sample_solver = None
server_sample_guide_scale = None
server_base_seed = None
server_offload_model = None
server_pipeline = None
outputs_dir = None

app = Flask(__name__)
job_queue = Queue()
jobs = {}

def worker():
    """Background worker processing one job at a time."""
    while True:
        job_id = job_queue.get()
        job = jobs[job_id]
        job['status'] = 'processing'
        try:
            prompt = job['prompt']
            img = Image.open(job['image_path']).convert('RGB')
            # Generate video
            video = server_pipeline.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[server_size],
                frame_num=job['frame_num'],
                shift=job['sample_shift'],
                sample_solver=job['sample_solver'],
                sampling_steps=job['sample_steps'],
                guide_scale=job['sample_guide_scale'],
                seed=job['base_seed'],
                offload_model=job['offload_model']
            )
            # Save result
            os.makedirs(outputs_dir, exist_ok=True)
            output_path = os.path.join(outputs_dir, f"{job_id}.mp4")
            cache_video(
                tensor=video[None],
                save_file=output_path,
                fps=server_cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            job['output_path'] = output_path
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)
        finally:
            job_queue.task_done()

# Start worker thread
worker_thread = Thread(target=worker, daemon=True)
worker_thread.start()

def str2bool(v):
    return str(v).lower() in ('true', '1', 'yes')

@app.route('/generate', methods=['POST'])
def generate_route():
    """Enqueue a video generation job."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    image_file = request.files['image']
    prompt = request.form.get('prompt', '')
    # Parse optional overrides
    try:
        frame_num = int(request.form.get('frame_num', server_frame_num))
        sample_steps = int(request.form.get('sample_steps', server_sample_steps))
        sample_shift = float(request.form.get('sample_shift', server_sample_shift))
        sample_solver = request.form.get('sample_solver', server_sample_solver)
        sample_guide_scale = float(request.form.get('sample_guide_scale', server_sample_guide_scale))
        base_seed = int(request.form.get('base_seed', server_base_seed))
        offload_model = str2bool(request.form.get('offload_model', server_offload_model))
    except Exception:
        return jsonify({'error': 'Invalid parameter value.'}), 400

    # Save input image
    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    job_id = uuid.uuid4().hex
    image_path = os.path.join(input_dir, f"{job_id}.png")
    image_file.save(image_path)

    # Create job record
    jobs[job_id] = {
        'status': 'queued',
        'prompt': prompt,
        'image_path': image_path,
        'frame_num': frame_num,
        'sample_steps': sample_steps,
        'sample_shift': sample_shift,
        'sample_solver': sample_solver,
        'sample_guide_scale': sample_guide_scale,
        'base_seed': base_seed,
        'offload_model': offload_model
    }
    job_queue.put(job_id)
    return jsonify({'job_id': job_id}), 202

@app.route('/status/<job_id>', methods=['GET'])
def status_route(job_id):
    """Return status of a generation job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found.'}), 404
    resp = {'status': job['status']}
    if job['status'] == 'done':
        resp['download_url'] = url_for('download_route', job_id=job_id, _external=True)
    elif job['status'] == 'error':
        resp['error'] = job.get('error', '')
    return jsonify(resp)

@app.route('/download/<job_id>', methods=['GET'])
def download_route(job_id):
    """Download the completed video."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found.'}), 404
    if job['status'] != 'done':
        return jsonify({'error': 'Job not completed.'}), 400
    return send_file(job['output_path'], as_attachment=True)

def main():
    global server_cfg, server_size, server_frame_num, server_sample_steps
    global server_sample_shift, server_sample_solver, server_sample_guide_scale
    global server_base_seed, server_offload_model, server_pipeline, outputs_dir

    parser = argparse.ArgumentParser(description='Wan I2V API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--task', type=str, default='i2v-14B', choices=list(WAN_CONFIGS.keys()), help='Wan task')
    parser.add_argument('--size', type=str, default='832*480', choices=list(SIZE_CONFIGS.keys()), help='Video size')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--frame_num', type=int, default=None, help='Number of frames')
    parser.add_argument('--sample_steps', type=int, default=None, help='Sampling steps')
    parser.add_argument('--sample_shift', type=float, default=None, help='Sampling shift')
    parser.add_argument('--sample_solver', type=str, default='unipc', choices=['unipc','dpm++'], help='Sampling solver')
    parser.add_argument('--sample_guide_scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--base_seed', type=int, default=-1, help='Base seed')
    parser.add_argument('--offload_model', type=str2bool, default=False, help='Offload model to CPU')
    parser.add_argument('--t5_fsdp', action='store_true', help='Use FSDP for T5')
    parser.add_argument('--dit_fsdp', action='store_true', help='Use FSDP for DiT')
    parser.add_argument('--t5_cpu', action='store_true', help='Place T5 on CPU')
    parser.add_argument('--ulysses_size', type=int, default=1, help='Ulysses parallel size')
    parser.add_argument('--ring_size', type=int, default=1, help='Ring attention parallel size')
    args = parser.parse_args()

    # Validate arguments
    assert args.task in WAN_CONFIGS, f'Unsupported task: {args.task}'
    assert os.path.isdir(args.ckpt_dir), 'Checkpoint directory must exist'

    # Set server-wide parameters
    server_cfg = WAN_CONFIGS[args.task]
    server_size = args.size
    server_frame_num = args.frame_num if args.frame_num is not None else (1 if 't2i' in args.task else 81)
    server_sample_steps = args.sample_steps if args.sample_steps is not None else (40 if 'i2v' in args.task else 50)
    server_sample_shift = args.sample_shift if args.sample_shift is not None else (3.0 if ('i2v' in args.task and args.size in ['832*480','480*832']) else 5.0)
    server_sample_solver = args.sample_solver
    server_sample_guide_scale = args.sample_guide_scale
    server_base_seed = args.base_seed
    server_offload_model = args.offload_model

    # Load model pipeline once
    device_id = int(os.getenv('LOCAL_RANK', 0))
    PipelineClass = wan.WanI2V if 'i2v' in args.task else wan.WanT2V
    server_pipeline = PipelineClass(
        config=server_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu
    )

    # Ensure outputs directory
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # Run Flask server
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()