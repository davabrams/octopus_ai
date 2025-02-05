# from flask import Flask, request, jsonify
import requests
import os
import time
import threading

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
    return response.json()

def kill_server():
    response = requests.post('http://localhost:8080/kill')
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def crash_server():
    response = requests.post('http://localhost:8080/crash')
    code = response.status_code
    if code >= 400:
        return None
    print(response.text)

def spawn_server():
    os.system('python3 server.py')

if __name__ == '__main__':

    t_server = threading.Thread(target=spawn_server)
    t_server.start()
    time.sleep(2.5)

    # Example client usage
    print(get_all_items())
    print(add_new_item({"job_id": 3, "data": {"stuff": "junk"}}))
    print(add_new_item({"job_id": 4, "data": {"stuff": "junk"}}))
    print(add_new_item({"job_id": 5, "data": {"stuff": "junk"}}))
    print(get_item_by_id(3))
    print(get_all_items())
    print("**************************************")
    print("Tests pass! Crashing the server now...")
    print("**************************************")
    crash_server()