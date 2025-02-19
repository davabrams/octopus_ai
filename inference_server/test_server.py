"""
Test for inference server

run using:
python3 test_server.py TestClientServer

"""
import time
import unittest
import requests
from server import app
import threading
import multiprocessing as mp

TIMEOUT = 0.1

def decode_response(r):
    encoding = r.encoding
    if encoding == "utf-8":
        return r.text
    else:
        print(encoding)
        return r.json()

# Python client to interact with the REST server
def get_all_items():
    response = requests.get('http://localhost:8080/list_jobs', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        print("Error", code)
        return None
    return response.json()

def get_item_by_id(item_id):
    response = requests.get(f'http://localhost:8080/jobs/{item_id}', timeout=TIMEOUT)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Item not found"}

def add_new_item(item):
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

def kill_server():
    response = requests.post('http://localhost:8080/kill', timeout=TIMEOUT)
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def shutdown_server():
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
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        shutdown_server()

    def test_communications(self):
        """
        tests major communication functionality
        """
        res = get_all_items()
        self.assertListEqual(res, [])
        add_new_item({"job_id": 3, "data": {"c.r": 0.52, "c_val.r": 1.0}})
        add_new_item({"job_id": 4, "data": {"c.r": 0.45, "c_val.r": 1.0}})
        add_new_item({"job_id": 5, "data": {"c.r": 0.32, "c_val.r": 1.0}})
        add_new_item({"job_id": 6, "data": {"c.r": 0.22, "c_val.r": 1.0}})
        item = get_item_by_id(3)
        self.assertDictEqual(item, {"job_id": "3","result": "None","status": "Pending"})
        res = get_all_items()
        self.assertEqual(len(res), 4)

        #let the jobs complete, then clear them
        time.sleep(3)
        item = get_item_by_id(3)
        self.assertDictEqual(item, {"job_id": "3","result": "0.74155515","status": "Complete"})
        collect_and_clear()
        res = get_all_items()
        self.assertListEqual(res, [])
        queues = show_queues()
        self.assertDictEqual(queues, {'completion_queue': [], 'execution_queue': [], 'pending_queue': []})

        

if __name__ == '__main__':
    unittest.main()

    list_threads_and_processes()
