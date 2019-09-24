""" Contains implementation of workitem class used in distributed RadIO system. """

from binascii import hexlify
import hashlib
import numpy as np


class Workitem(object):
    def __init__(self, client_id, scan_path, fmt='dicom',
                 config=None, timeout=20 * 60):
        self.client_id = client_id
        self.item_id = self.generate_hash()
        self.scan_path = scan_path
        self.fmt = fmt
        self.processed_by = None
        self.predicted_nodules = None
        self.predicted_mask = None
        self.config = {} if config is None else dict(config)
        self.time_put = None
        self.timeout = timeout

    @staticmethod
    def generate_hash():
        sha256_res = hashlib.sha256(hexlify(np.random.rand(5)))
        result_hash = sha256_res.hexdigest()
        return result_hash

    def __str__(self):
        return "<Workitem id=%s>" % str(self.item_id)

    def __eq__(self, other):
        return self.item_id == other.item_id

    def __hash__(self):
        return hash(self.item_id)
