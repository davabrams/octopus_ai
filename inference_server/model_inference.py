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
        try:
            res = sucker_model.predict(np.array([[c, c_val]]), verbose=0)[0][0]
            self.parent.result = res
            self.parent.status = JobStatus.COMPLETE
        except Exception as e:
            logging.error("Error in inference: %s", e)
            self.parent.status = JobStatus.FAILED


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
        assert(self.status in (JobStatus.COMPLETE , JobStatus.FAILED))


        logging.info(
            "%s started at %s compelted at %s (thread %s), resolution: %s",
            threading.current_thread().name,
            self.inference_start_timestamp,
            self.inference_end_timestamp,
            threading.get_ident(),
            self.status
        )
        if self.status == JobStatus.COMPLETE:
            parent_queue.move_to_complete(self.job_id)
        else:
            logging.error("Job failed, not writing to completion queue")


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
    _kill_watchdog: threading.Event

    _watchdog_thread = None

    def __init__(self) -> None:
        # Start the queue watchdog
        logging.info("Starting Watchdog Thread")
        self._kill_watchdog = threading.Event()
        self._kill_watchdog.clear()
        if self.is_watchdog_alive():
            logging.error("A watchdog is already alive on %s", self._watchdog_thread.native_id)
            return
        t1 = threading.Thread(
            target=self.queue_watchdog, args=(self._kill_watchdog,), name="InferenceQueue Watchdog",
            
        )
        t1.daemon = True
        t1.start()
        self._watchdog_thread = t1
        logging.info("Spawned watchdog thread %s", t1.name)

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
        logging.info("Deleting %s", job_id)
        del self._q[job_id]
        self._pending_queue = rem_job_id(self._pending_queue)
        self._execution_queue = rem_job_id(self._execution_queue)
        self._completion_queue = rem_job_id(self._completion_queue)

    def reset_all_queues(self) -> None:
        self._pending_queue = []
        self._execution_queue = []
        self._completion_queue = []
        self._q = {}
        self._ts_index = None
        self._q_ptr = None

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
    
    def kill_watchdog(self) -> None:
        """
        Kills the watchdog thread
        """
        logging.warning("Kill signal received")
        self._kill_watchdog.set()

    # Internal commands for watchdog and executions
    def clear_stale(self) -> None:
        """
        For now, only clear stale pending jobs
        """
        now = time.time()
        cutoff = now + self.seconds_until_stale
        job_ids_for_removal = [ts[1] for ts in self._pending_queue if ts[0] < cutoff]
        if len(job_ids_for_removal) == 0:
            return []
        logging.warning("Removing expired pending jobs: %s", ",".join([str(i) for i in job_ids_for_removal]))
        for id_to_remove in job_ids_for_removal:
            del self._q[id_to_remove]
        self._pending_queue = [ts for ts in self._pending_queue if ts[1] not in job_ids_for_removal]

    def execute_new_job(self) -> None:
        """
        1) Find the most recent job
        2) Move it to execution
        3) Kick off its execution
        """
        if self._kill_watchdog.is_set():
            logging.error("Kill watchdog flag is set, aborting execution")
            return
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

    def destroy_job(self, job_id: int) -> None:
        """
        Move a job out of the execution queue, but not into the complete queue
        """
        self._execution_queue = [
            job for job in self._execution_queue if job[1] is not job_id
        ]

    def queue_watchdog(self, kill_watchdog: threading.Event) -> None:
        """
        Watchdog for the inference queue.  Looks for new jobs to execute, cleans old jobs, etc.
        """
        logging.info("Watchdog is starting on thread %s", threading.current_thread)
        logging.info("Thread ID: %s", threading.get_ident())
        logging.info("Thread Name: %s", threading.current_thread().name)
        while True:
            if kill_watchdog.is_set():
                return
            time.sleep(0.1)
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

    def is_watchdog_alive(self) -> bool:
        """
        Check if the watchdog thread is alive
        """
        if self._watchdog_thread is None:
            return False
        return self._watchdog_thread.is_alive()