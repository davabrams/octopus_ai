"""
Run model inference
"""
import sys
import logging
import time
import datetime
import threading
from typing import Any
from enum import StrEnum
from abc import ABC
from tensorflow import keras
import numpy as np

sys.path.insert(1, '..')
from training.losses import ConstraintLoss
from OctoConfig import default_models, MLMode

logging.basicConfig(level=logging.INFO)
logging.info("The log level for this message is INFO.")

model_path = default_models[MLMode.SUCKER]
custom_objects = {"ConstraintLoss": ConstraintLoss}
sucker_model = keras.models.load_model('../' + model_path, custom_objects)


class JobStatus(StrEnum):
    """A job can be one of four statues"""
    PENDING = "Pending"
    EXECUTING = "Executing"
    COMPLETE = "Complete"
    FAILED = "Failed"


class InferenceExecutionBase(ABC):
    """Base class for execution"""
    def __init__(self, parent: Any, data: Any):
        self.parent = parent
        self.data = data
        self.execute()

    def execute(self):
        """Placeholder execution function"""
        raise NotImplementedError


class ExecuteSuckerInference(InferenceExecutionBase):
    """Sucker execution class"""
    def execute(self) -> float:
        print(self.data)
        c = self.data["c.r"]
        c_val = self.data["c_val.r"]
        logging.info(
            "%s Executing (Thread ID: %s)",
            threading.current_thread().name,
            threading.get_ident(),
        )
        res = sucker_model.predict(np.array([[c, c_val]]), verbose=0)[0][0]
        self.parent.result = res


class InferenceJob:
    """
    Stores an inference task
    """

    result = None
    queue_timestamp = None
    inference_start_timestamp = None
    inference_end_timestamp = None
    execution_binding = None

    def __init__(self, item_details: dict) -> None:
        self.job_id = int(item_details["job_id"])
        self.data = item_details["data"]
        self.status = JobStatus.PENDING
        self.queue_timestamp = time.time()

    def __repr__(self) -> str:
        return f"{self.job_id}: {self.status}"

    def process(self) -> None:
        """
        Process inference
        """
        raise NotImplementedError

    def as_json(self):
        """
        Return contents of job status and result as json object
        """
        result_dict = {
            "job_id": str(self.job_id),
            "status": str(self.status),
            "result": str(self.result),
        }
        return result_dict

    def get_human_readable_timestamps(self):
        """
        Gets all the timing information for a job
        """
        res = {
            "queue_timestamp": None,
            "inference_start_timestamp": None,
            "inference_end_timestamp": None,
        }
        if self.queue_timestamp:
            res["queue_timestamp"] = datetime.datetime.fromtimestamp(
                self.queue_timestamp
            ).isoformat()
        return res

    def execute(self, parent_queue: Any):
        """Kick off inference, this belongs in its own thread"""
        self.inference_start_timestamp = time.time()
        self.status = JobStatus.EXECUTING

        ExecuteSuckerInference(parent=self, data=self.data)

        self.inference_end_timestamp = time.time()
        self.status = JobStatus.COMPLETE  # a little optimistic? lol

        logging.info(
            "%s started at %s compelted at %s (thread %s)",
            threading.current_thread().name,
            self.inference_start_timestamp,
            self.inference_end_timestamp,
            threading.get_ident(),
        )
        parent_queue.move_to_complete(self.job_id)


class InferenceQueue:
    """
    A queue

    add : user adds new jobs, and they go into the pending queue
    get : user gets a job by job_id

    delete : delete a job by job_id
    execute_new_job : moves a job out of the pending queue and into
        the execution queue and executes it
    """

    thread_count = 2
    seconds_until_stale = 30.0

    _q_ptr = None
    _q = {}  # job_id : job .  these don't move, things just point to them.

    # (timestamp, job_id), kept in ascending order.  should be a tree or something.
    _pending_queue = []
    _execution_queue = []
    _completion_queue = []
    _ts_index = None
    _keep_alive: bool = True

    def __init__(self) -> None:
        # Start the queue watchdog
        logging.info("Starting Watchdog Thread")
        t1 = threading.Thread(
            target=self.queue_watchdog, name="InferenceQueue Watchdog"
        )
        t1.start()
        logging.info("Spawned watchdog thread %s", t1.getName())

    # End user commands:
    def add(self, job: InferenceJob) -> None:
        """
        Adds to the pending queue
        """
        logging.info("Adding %s", job.job_id)
        self._q[job.job_id] = job
        self._pending_queue.append((job.queue_timestamp, job.job_id))

    def get(self, job_id: int) -> InferenceJob:
        """Return a job by ID"""
        if job_id not in self._q:
            raise IndexError
        return self._q[job_id]

    def delete(self, job_id: int) -> None:
        """Delete a job by ID"""
        def rem_job_id(queue):
            return [elem for elem in queue if elem[1] is not job_id]

        if job_id not in self._q:
            raise IndexError
        del self._q[job_id]
        self._pending_queue = rem_job_id(self._pending_queue)
        self._execution_queue = rem_job_id(self._execution_queue)
        self._completion_queue = rem_job_id(self._completion_queue)

    def all_job_ids(self) -> list:
        """Return all job IDs still in the queue"""
        return list(self._q.keys())

    def all_queues(self) -> dict:
        """Return all job queues"""
        return {
            "pending_queue": self._pending_queue,
            "execution_queue": self._execution_queue,
            "completion_queue": self._completion_queue,
        }

    def collect_and_clear(self) -> list:
        """Gather all compelted jobs, return them, and erase them"""
        res = []
        for q_elem in self._completion_queue:
            job_id = q_elem[1]
            job = self._q[job_id]
            job_json = job.as_json()
            res.append(job_json)
            del self._q[job_id]
        self._completion_queue = []
        return res

    def __iter__(self):
        # Iterates in chronological order
        self._q_ptr = 0
        self._ts_index = (
            self._pending_queue + self._execution_queue + self._completion_queue
        )
        return self

    def __next__(self):
        # Iterates in chronological order
        ptr = self._q_ptr
        if ptr >= len(self._ts_index):
            raise StopIteration
        self._q_ptr += 1
        job_id = self._ts_index[ptr][1]
        return self._q[job_id]
    
    def kill_queue(self) -> None:
        logging.warning("Kill signal received")
        self._keep_alive = False

    # Internal commands for watchdog and executions
    def clear_stale(self) -> None:
        """
        For now, only clear stale pending jobs
        """
        def prune(ts_list: list) -> list:
            return [ts for ts in ts_list if ts[0] > cutoff]

        now = time.time()
        cutoff = now + self.seconds_until_stale
        logging.info("Cutoff time is %s", cutoff)
        self._pending_queue = prune(self._pending_queue)

    def execute_new_job(self) -> None:
        """
        1) Find the most recent job
        2) Move it to execution
        3) Kick off its execution
        """
        # 1
        logging.info("Executing new job")
        if len(self._pending_queue) == 0:
            return None
        job_ts, job_id = self._pending_queue[-1]

        # 2
        self._execution_queue.append((job_ts, job_id))
        self._q[job_id].status = JobStatus.EXECUTING
        del self._pending_queue[-1]

        # 3
        logging.info("Kicking off thread")
        t1 = threading.Thread(
            target=self._q[job_id].execute(self), name=f"Executor for {job_id}"
        )
        self._q[job_id].execution_binding = t1
        t1.start()

    def move_to_complete(self, job_id: int) -> None:
        """
        Move a job from execution queue to complete queue
        """
        job_singlet = [job for job in self._execution_queue if job[1] == job_id]
        if len(job_singlet) != 1:
            logging.warning("How did this even happen?")
            return
        job = job_singlet[0]
        self._execution_queue = [
            job for job in self._execution_queue if job[1] is not job_id
        ]
        self._completion_queue.append(job)

    def queue_watchdog(self) -> None:
        """
        Watchdog for the inference queue.  Looks for new jobs to execute, cleans old jobs, etc.
        """
        logging.info("Watchdog is starting on thread %s", threading.current_thread)
        logging.info("Thread ID: %s", threading.get_ident())
        logging.info("Thread Name: %s", threading.current_thread().name)

        while self._keep_alive:
            time.sleep(0.5)
            # self.clear_stale()
            if (
                len(self._pending_queue)
                or len(self._execution_queue)
                or len(self._completion_queue)
            ):
                logging.info(
                    "%s %s %s",
                    len(self._pending_queue),
                    len(self._execution_queue),
                    len(self._completion_queue),
                )
            while (
                len(self._pending_queue) > 0
                and len(self._execution_queue) <= self.thread_count
            ):
                self.execute_new_job()
