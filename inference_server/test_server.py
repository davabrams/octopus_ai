"""
Test for inference server

run using:
python3 test_server.py

note, the following should work but it loads two watchdogs:
python3 test_server.py TestClientServer

"""
import time
import unittest
import requests
from server import app
import threading
import multiprocessing as mp
from model_inference import InferenceJob, InferenceQueue

TIMEOUT = 0.1

def decode_response(r):
    encoding = r.encoding
    if encoding == "utf-8":
        return r.text
    else:
        print(encoding)
        return r.json()

# Python client to interact with the REST server
def list_jobs():
    response = requests.get('http://localhost:8080/list_jobs', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        print("Error", code)
        return None
    return response.json()

def job_by_id(item_id):
    response = requests.get(f'http://localhost:8080/jobs/{item_id}', timeout=TIMEOUT)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Item not found"}

def add_job(item):
    response = requests.post('http://localhost:8080/jobs', json=item, timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        return None
    return decode_response(response)

def show_queues():
    response = requests.get('http://localhost:8080/show_queues', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        return None
    return response.json()

def shutdown():
    print("Shutting down server")
    response = requests.get('http://localhost:8080/shutdown', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        return f"Error {code}"
    print(response.text)

def collect_and_clear():
    response = requests.post('http://localhost:8080/collect_and_clear', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def list_threads_and_processes() -> None:
    """
    List active threads and processes
    """
    for ix, thread in enumerate(threading.enumerate()):
        print(f"Thread {ix}: {thread.name}")
    for ix, proc in enumerate(mp.active_children()):
        print(f"Process {ix}: {proc.name}")

class TestClientServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t1 = threading.Thread(target=lambda: app.run(host='localhost',port=8080, debug=True, use_reloader=False), name="REST Server")
        t1.daemon = True
        t1.start()

    @classmethod
    def tearDownClass(cls):
        shutdown()

    def test_communications(self):
        """
        tests major communication functionality
        """
        res = list_jobs()
        self.assertListEqual(res, [])
        add_job({"job_id": 3, "data": {"c.r": 0.52, "c_val.r": 1.0}})
        add_job({"job_id": 4, "data": {"c.r": 0.45, "c_val.r": 1.0}})
        add_job({"job_id": 5, "data": {"c.r": 0.32, "c_val.r": 1.0}})
        res = list_jobs()
        self.assertEqual(len(res), 3)
        add_job({"job_id": 5, "data": {"c.r": 0.22, "c_val.r": 1.0}})
        item = job_by_id(3)
        self.assertDictEqual(item, {"job_id": "3","result": "None","status": "Pending"})
        res = list_jobs()
        self.assertEqual(len(res), 3)

        #let the jobs complete, then clear them
        time.sleep(.5)
        item = job_by_id(3)
        self.assertDictEqual(item, {"job_id": "3","result": "0.74155515","status": "Complete"})
        collect_and_clear()
        res = list_jobs()
        self.assertListEqual(res, [])
        queues = show_queues()
        self.assertDictEqual(queues, {'completion_queue': [], 'execution_queue': [], 'pending_queue': []})

class TestInferenceServer(unittest.TestCase):
    """
    Tests the queue functionality
    """
    def test_inference_jobs(self):
        job_details = {
            "job_id": 123,
            "data": "hellow",
        }
        job = InferenceJob(job_details)
        self.assertDictEqual(
            job.as_json(),
            {
                "job_id": "123",
                "status": "Pending",
                "result": "None"
            }
        )

    def test_inference_queue(self):
        q = InferenceQueue()
        q.reset_all_queues()
        # Stop the watchdog so we can test the queue.
        self.assertFalse(q._kill_watchdog.is_set())
        q.kill_watchdog()
        self.assertTrue(q._kill_watchdog.is_set())
        time.sleep(0.1)
        ids = q.all_job_ids()
        self.assertListEqual(ids, [])
        queues = q.all_queues()
        self.assertDictEqual(
            queues,
            {
            "pending_queue": [],
            "execution_queue": [],
            "completion_queue": []
            }
        )

        for i in range(5):
            job_details = {
                "job_id": str(i),
                "data": {"contents": str(i*2)}
            }
            q.add(InferenceJob(job_details))

        self.assertListEqual(q.all_job_ids(), [i for i in range(5)])
        queues = q.all_queues()
        self.assertEqual(len(queues["pending_queue"]), 5)
        self.assertEqual(len(queues["execution_queue"]), 0)
        self.assertEqual(len(queues["completion_queue"]), 0)

        # Test iterator
        for i, j in enumerate(q):
            self.assertEqual(j.job_id, i)

        self.assertListEqual(q.all_job_ids(), list(range(5)))

        for i in range(5):
            q.delete(i)
            queues = q.all_queues()
            self.assertEqual(len(queues["pending_queue"]), 4-i)
        self.assertEqual(len(q.all_job_ids()), 0)

        # Test the clear stale function
        q.seconds_until_stale = 0.1
        job_details = {
            "job_id": 100,
            "data": None
        }
        q.add(InferenceJob(job_details))
        self.assertEqual(len(q.all_job_ids()), 1)

        time.sleep(0.2)
        q.clear_stale()
        self.assertEqual(len(q.all_job_ids()), 0)


if __name__ == '__main__':
    unittest.main()

    list_threads_and_processes()
