#!/usr/bin/python
""" Contains implementation dispatcher class used in distributed RadIO framework. """
import os
import time
from collections import defaultdict
import socket
import logging
import argparse
from datetime import datetime
from threading import Thread
from multiprocessing import Queue
from multiprocessing.queues import Empty
import Pyro4
from radio.utils import DEFAULT_CONFIG as CONFIG


Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
Pyro4.config.HOST = socket.gethostbyname('dispatcher')


class RadIODispatcher:
    def __init__(self, heartbeat_timeout=0.005, sleep_time=0.2):
        self.logger = logging.getLogger('RadIO.heartbeat')

        self.work_queue = Queue()
        self.processing_queue = defaultdict(Queue)
        self.result_queue = defaultdict(Queue)

        self.last_heartbeat = defaultdict(float)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.sleep_time = float(sleep_time)

        self._heartbeat_thread = Thread(target=self.heartbeat_check)
        self.logger.info("Starting heartbeat check thread...")
        self._heartbeat_thread.start()

    def restart_worker_jobs(self, worker_id: str):
        worker_processing_queue = self.processing_queue[worker_id]
        self.logger.info("Restarting all worker {} jobs...".format(worker_id))
        i = 0
        while worker_processing_queue.qsize() != 0:
            current_item = worker_processing_queue.get()
            current_item.procesed_by = None
            self.work_queue.put(current_item)
            i += 1
        del self.last_heartbeat[worker_id]

    def heartbeat_check(self):
        while True:
            self.logger.info(self.last_heartbeat)
            workers_list = list(self.last_heartbeat.keys())
            for worker_id in workers_list:
                current_time = time.clock()
                prev_heartbeat = self.last_heartbeat.get(worker_id, None)
                if prev_heartbeat is None:
                    continue
                heartbeat_delta = current_time - prev_heartbeat
                if heartbeat_delta > self.heartbeat_timeout:
                    self.restart_worker_jobs(worker_id)
            time.sleep(self.sleep_time)

    @Pyro4.expose
    def heartbeat(self, worker_id: str):
        self.last_heartbeat[worker_id] = time.clock()
        self.logger.info(self.last_heartbeat)
        return True

    @Pyro4.expose
    def put_work(self, item: 'WorkItem'):
        logger = logging.getLogger('RadIO.dispatcher')
        item.time_put = datetime.now()
        self.work_queue.put(item)
        logger.info("Putting worker item: {}".format(item))
        return True

    @Pyro4.expose
    def get_work(self, worker_id: str, timeout: float = 5):
        logger = logging.getLogger('RadIO.dispatcher')
        logger.info("Getting item from work queue...")
        item = self.work_queue.get(block=True)
        logger.info("Got item {}".format(item))
        item.processed_by = worker_id
        self.processing_queue[worker_id].put(item)
        logger.info("Current state of processing queue: {}".format(
            self.processing_queue[worker_id].qsize()))
        return item

    @Pyro4.expose
    def put_result(self, worker_id: str, item: 'WorkItem'):
        logger = logging.getLogger('RadIO.dispatcher')
        logger.info(
            "Getting processing queue for worker {}".format(worker_id))
        current_processing_queue = self.processing_queue[worker_id]

        logger.info(
            "Getting result queue for client {}".format(item.client_id))
        current_result_queue = self.result_queue[item.client_id]
        logger.info("Getting {} from {} queue".format(item, worker_id))
        _ = current_processing_queue.get()  # noqa: F841
        logger.info("Successfuly remove item from curret processing queue..")
        logger.info("Putting item into result queue...")
        if item.time_put is None or item.timeout is None:
            current_result_queue.put(item)
        elif item.timeout < (datetime.now() - item.time_put).total_seconds():
            logger.info("Timeout for prediction was exceeded " +
                        " for item {}.".format(item.item_id))
            return True
        else:
            current_result_queue.put(item)
        logger.info("Successfuly put item into result queue...")
        return True

    @Pyro4.expose
    def get_result(self, client_id: str, timeout: float = None):
        logger = logging.getLogger('RadIO.dispatcher')
        logger.info("Getting result for client '{}': ".format(client_id) +
                    "size of result queue is {}".format(self.get_result_qsize(client_id)))
        current_result_queue = self.result_queue[client_id]
        try:
            logger.info("Trying to get prediction: " +
                        " timeout is {}".format(timeout))
            result = current_result_queue.get(timeout=timeout)
        except Empty as e:
            logger.error("Queue is Empty error happened. " +
                         "It seams that timeout is exceeded.")
            result = e
        except Exception as e:
            logger.error("Unknown error happened when " +
                         "trying to get result for client {}. ".format(client_id) +
                         "Error: {}. ".format(e) +
                         "Returning 'None' value..."
                         )
            result = None
        return result

    @Pyro4.expose
    def get_work_qsize(self) -> int:
        return self.work_queue.qsize()

    @Pyro4.expose
    def get_result_qsize(self, client_id: str) -> int:
        return self.result_queue[client_id].qsize()

    @Pyro4.expose
    def get_worker_qsize(self, worker_id: str) -> int:
        logger = logging.getLogger('RadIO.dispatcher')
        logger.info("Getting size of processing " +
                    " queue for worker '{}'...".format(worker_id))
        return self.processing_queue[worker_id].qsize()

    @Pyro4.expose
    def get_worker_ids(self) -> 'List[str]':
        logger = logging.getLogger('RadIO.dispatcher')
        logger.info("Getting all workers ids...")
        return list(self.last_heartbeat.keys())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "This script starts dispatcher that will handle requests for prediction" +
        " and distribute them among active workers.")
    )

    args = parser.parse_args()

    os.system("python -m Pyro4.naming -n dispatcher -p 8888 " +
              "> /radio/nameserver.log &")
    logger = logging.getLogger('RadIO.dispatcher')

    logger.info("Creating dispatcher object...")
    logger.info("Sleep time is set to {}".format(CONFIG
                                                 .deploy
                                                 .distributed
                                                 .sleep_time))
    logger.info("Heartbeat timeout is set to {}".format(CONFIG
                                                        .deploy
                                                        .distributed
                                                        .heartbeat_timeout))
    dispatcher = RadIODispatcher(
        sleep_time=CONFIG.deploy.distributed.sleep_time,
        heartbeat_timeout=CONFIG.deploy.distributed.heartbeat_timeout
    )
    logger.info("Dispatcher is running " +
                "on {}".format(socket.gethostbyname('dispatcher')))
    Pyro4.Daemon.serveSimple(
        host=socket.gethostbyname('dispatcher'),
        objects={dispatcher: 'dispatcher'}, ns=True
    )
