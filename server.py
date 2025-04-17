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
# Optional TeaCache acceleration
try:
    from teacache_generate import t2v_generate, i2v_generate, teacache_forward
    _TEACACHE_AVAILABLE = True
except ImportError:
    _TEACACHE_AVAILABLE = False
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Supabase client initialization
try:
    from supabase import create_client
except ImportError:
    logger.error("supabase module not found, please install supabase>=1.0.0")
    raise

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase environment variables SUPABASE_URL and SUPABASE_KEY must be set")
    raise RuntimeError("Supabase environment variables SUPABASE_URL and SUPABASE_KEY must be set")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    logger.info("Worker thread started, waiting for jobs.")
    while True:
        job_id = job_queue.get()
        logger.info(f"Worker picked up job {job_id}")
        job = jobs[job_id]
        job['status'] = 'processing'
        logger.info(f"Job {job_id}: status set to 'processing'")
        try:
            prompt = job['prompt']
            logger.info(f"Job {job_id}: prompt='{prompt}'")
            logger.info(f"Job {job_id}: loading image from {job['image_path']}")
            img = Image.open(job['image_path']).convert('RGB')
            logger.info(f"Job {job_id}: image loaded successfully")
            # Generate video
            logger.info(f"Job {job_id}: starting video generation with frame_num={job['frame_num']}, sample_steps={job['sample_steps']}, sample_shift={job['sample_shift']}, sample_solver={job['sample_solver']}, guide_scale={job['sample_guide_scale']}, base_seed={job['base_seed']}, offload_model={job['offload_model']}")
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
            logger.info(f"Job {job_id}: video generation completed")
            # Save result
            logger.info(f"Job {job_id}: ensuring output directory {outputs_dir}")
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
            # Upload generated video to Supabase storage
            try:
                with open(output_path, "rb") as video_file:
                    video_data = video_file.read()
                res = supabase.storage.from_("videos").upload(f"{job_id}.mp4", video_data, {"contentType": "video/mp4"})
                if res.get("error"):
                    logger.error(f"Supabase video upload error for job {job_id}: {res['error']}")
                else:
                    video_url = supabase.storage.from_("videos").get_public_url(f"{job_id}.mp4")["publicUrl"]
                    supabase.table("Video").update({"videoUrl": video_url, "status": "succeeded"}).eq("id", job_id).execute()
            except Exception as e:
                logger.error(f"Error uploading video to Supabase for job {job_id}: {e}", exc_info=True)
            job['status'] = 'done'
            logger.info(f"Job {job_id}: completed successfully, output at {output_path}")
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)
            logger.error(f"Job {job_id}: error during processing: {e}", exc_info=True)
        finally:
            job_queue.task_done()
            logger.info(f"Job {job_id}: task done, remaining queue size {job_queue.qsize()}")

# Start worker thread
worker_thread = Thread(target=worker, daemon=True)
worker_thread.start()

def str2bool(v):
    return str(v).lower() in ('true', '1', 'yes')

@app.route('/generate', methods=['POST'])
def generate_route():
    """Enqueue a video generation job."""
    logger.info(f"Received /generate request from {request.remote_addr}")
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    image_file = request.files['image']
    prompt = request.form.get('prompt', '')
    logger.info(f"/generate: prompt='{prompt}'")
    # Validate user ID
    user_id = request.form.get('userId')
    if not user_id:
        logger.error("/generate: no userId provided")
        return jsonify({'error': 'No userId provided.'}), 400
    # Parse optional overrides
    try:
        frame_num = int(request.form.get('frame_num', server_frame_num))
        sample_steps = int(request.form.get('sample_steps', server_sample_steps))
        sample_shift = float(request.form.get('sample_shift', server_sample_shift))
        sample_solver = request.form.get('sample_solver', server_sample_solver)
        sample_guide_scale = float(request.form.get('sample_guide_scale', server_sample_guide_scale))
        base_seed = int(request.form.get('base_seed', server_base_seed))
        offload_model = str2bool(request.form.get('offload_model', server_offload_model))
        logger.info(f"/generate: parameters frame_num={frame_num}, sample_steps={sample_steps}, sample_shift={sample_shift}, sample_solver={sample_solver}, sample_guide_scale={sample_guide_scale}, base_seed={base_seed}, offload_model={offload_model}")
    except Exception:
        logger.error("/generate: invalid parameter value", exc_info=True)
        return jsonify({'error': 'Invalid parameter value.'}), 400

    # Save input image
    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    job_id = uuid.uuid4().hex
    image_path = os.path.join(input_dir, f"{job_id}.png")
    image_file.save(image_path)
    logger.info(f"/generate: saved input image to {image_path} for job {job_id}")

    # Upload input image to Supabase storage
    try:
        with open(image_path, "rb") as img_f:
            image_bytes = img_f.read()
        res = supabase.storage.from_("images").upload(f"{job_id}.png", image_bytes, {"contentType": "image/png"})
        if res.get("error"):
            logger.error(f"Supabase image upload error for job {job_id}: {res['error']}")
            return jsonify({'error': 'Failed to upload image to Supabase.'}), 500
        image_url = supabase.storage.from_("images").get_public_url(f"{job_id}.png")["publicUrl"]
    except Exception as e:
        logger.error(f"Error uploading image to Supabase for job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Error uploading image.'}), 500

    # Create Video record in Supabase database
    try:
        record = {
            "id": job_id,
            "userId": user_id,
            "prompt": prompt,
            "imageUrl": image_url,
            "status": "processing"
        }
        supabase.table("Video").insert(record).execute()
    except Exception as e:
        logger.error(f"Error inserting Video record for job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create video record.'}), 500

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
    logger.info(f"/generate: enqueued job {job_id}")
    return jsonify({'job_id': job_id}), 202

@app.route('/status/<job_id>', methods=['GET'])
def status_route(job_id):
    """Return status of a generation job."""
    logger.info(f"Received /status request from {request.remote_addr} for job {job_id}")
    job = jobs.get(job_id)
    if not job:
        logger.warning(f"/status: job {job_id} not found")
        return jsonify({'error': 'Job not found.'}), 404
    resp = {'status': job['status']}
    if job['status'] == 'done':
        resp['download_url'] = url_for('download_route', job_id=job_id, _external=True)
    elif job['status'] == 'error':
        resp['error'] = job.get('error', '')
    logger.info(f"/status: job {job_id} status is {job['status']}")
    return jsonify(resp)

@app.route('/download/<job_id>', methods=['GET'])
def download_route(job_id):
    """Download the completed video."""
    logger.info(f"Received /download request from {request.remote_addr} for job {job_id}")
    job = jobs.get(job_id)
    if not job:
        logger.warning(f"/download: job {job_id} not found")
        return jsonify({'error': 'Job not found.'}), 404
    if job['status'] != 'done':
        logger.warning(f"/download: job {job_id} not completed, status {job['status']}")
        return jsonify({'error': 'Job not completed.'}), 400
    logger.info(f"/download: sending file {job['output_path']} for job {job_id}")
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
    # TeaCache options
    parser.add_argument('--enable_teacache', action='store_true', default=False,
                        help='Enable TeaCache caching to accelerate sampling')
    parser.add_argument('--teacache_thresh', type=float, default=0.2,
                        help='Threshold for TeaCache L1 distance to trigger compute')
    parser.add_argument('--use_ret_steps', action='store_true', default=False,
                        help='Use retention steps strategy in TeaCache')
    args = parser.parse_args()
    logger.info(f"Starting server with: host={args.host}, port={args.port}, task={args.task}, size={args.size}, ckpt_dir={args.ckpt_dir}, frame_num={args.frame_num}, sample_steps={args.sample_steps}, sample_shift={args.sample_shift}, sample_solver={args.sample_solver}, sample_guide_scale={args.sample_guide_scale}, base_seed={args.base_seed}, offload_model={args.offload_model}, t5_fsdp={args.t5_fsdp}, dit_fsdp={args.dit_fsdp}, t5_cpu={args.t5_cpu}, ulysses_size={args.ulysses_size}, ring_size={args.ring_size}")

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
    logger.info(f"Loaded pipeline {PipelineClass.__name__} with checkpoint_dir={args.ckpt_dir}, device_id={device_id}, t5_fsdp={args.t5_fsdp}, dit_fsdp={args.dit_fsdp}, use_usp={args.ulysses_size > 1 or args.ring_size > 1}, t5_cpu={args.t5_cpu}")

    # Ensure outputs directory
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    logger.info(f"Outputs directory is {outputs_dir}")

    # Run Flask server
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()