"""
Octopus model inference server.  
Originally done in C++ but tensorflow in C++ was too much of a pain.
"""
import logging
from flask import Flask, request, jsonify
from model_inference import InferenceJob, InferenceQueue
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logging.info('The log level for this message is INFO.')

jobs = InferenceQueue()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    """Server shutdown.  Does not kill the process."""
    jobs.kill_watchdog()
    return 'Server shutting down...', 201

@app.route('/list_jobs', methods=['GET'])
def get_items():
    """
    List all jobs and their status from the octopus ML server
    """
    job_status_queue = []
    for job in iter(jobs):
        job_status_queue.append({job.job_id: job.status})
    res = jsonify(job_status_queue)
    logging.info("job_status_queue: %s", res)
    return res, 201

@app.route('/show_queues', methods=['GET'])
def show_queues():
    """Expose all three job queues"""
    return jobs.all_queues(), 201

@app.route('/jobs', methods=['POST'])
def add_item():
    """
    Receive requests for model inference
    """
    new_item = request.get_json()
    if new_item["job_id"] in jobs.all_job_ids():
        return f"Job {new_item['job_id']} Exists", 500
    jobs.add(InferenceJob(new_item))
    return f"Added Job {new_item['job_id']}", 201

@app.route('/collect_and_clear', methods=['POST'])
def collect_and_clear():
    """Collect completed tasks from completed queue and clear them"""
    logging.warning("Collecting and clearing...")
    res = jobs.collect_and_clear()
    logging.warning("%s jobs collected and cleared", len(res))
    return res, 201

@app.route('/jobs/<int:job_id>', methods=['GET'])
def get_item(job_id):
    """
    Get a specific result for a job based on job ID
    """
    if job_id not in jobs.all_job_ids():
        res = ",".join([str(job) for job in jobs.all_job_ids()])
        return f"Item not found.  Items: {res}", 404
    job = jobs.get(job_id)
    res = job.as_json()
    if job.status == "Success":
        jobs.delete(job_id)
        return res, 201
    if job.status == "Failure":
        jobs.delete(job_id)
        return res, 500
    return res, 200

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
