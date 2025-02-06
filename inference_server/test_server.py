"""
Test for inference server

run using:
python3 server.py & python3 test_server.py

"""
import time
import requests
from server import app

def decode_response(r):
    encoding = r.encoding
    if encoding == "utf-8":
        return r.text
    else:
        print(encoding)
        return r.json()

# Python client to interact with the REST server
def get_all_items():
    response = requests.get('http://localhost:8080/list_jobs')
    code = response.status_code
    if code >= 400:
        print("Error", code)
        return None
    return response.json()

def get_item_by_id(item_id):
    response = requests.get(f'http://localhost:8080/jobs/{item_id}')
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Item not found"}

def add_new_item(item):
    response = requests.post('http://localhost:8080/jobs', json=item)
    code = response.status_code
    if code >= 400:
        return None
    return decode_response(response)

def show_queues():
    response = requests.get('http://localhost:8080/show_queues')
    code = response.status_code
    if code >= 400:
        return None
    return response.json()


def kill_server():
    response = requests.post('http://localhost:8080/kill')
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def shutdown_server():
    response = requests.post('http://localhost:8080/shutdown')
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def collect_and_clear():
    response = requests.post('http://localhost:8080/collect_and_clear')
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)


if __name__ == '__main__':
    # Client usage
    time.sleep(2.5)
    print(get_all_items())
    print(add_new_item({"job_id": 3, "data": {"c.r": 0.52, "c_val.r": 1.0}}))
    print(show_queues())
    print(add_new_item({"job_id": 4, "data": {"c.r": 0.45, "c_val.r": 1.0}}))
    print(show_queues())
    print(add_new_item({"job_id": 5, "data": {"c.r": 0.32, "c_val.r": 1.0}}))
    print(show_queues())
    print(add_new_item({"job_id": 5, "data": {"c.r": 0.22, "c_val.r": 1.0}}))
    print(show_queues())
    time.sleep(2.5)
    print(get_item_by_id(3))
    time.sleep(2.5)
    print(get_all_items())
    print(show_queues())
    print(collect_and_clear())
    print(get_all_items())
    print(show_queues())
    shutdown_server()
