from models.pytorch.estimator import Estimator
from models.pytorch.estimator import *
from models.pytorch.metrics import as_torch as _as_torch, as_numpy as _as_numpy

class NewEstimator(Estimator):
    def __init__(self, model, save_folder, cuda_device=None, loss_fn=None, optimizer=None):
        if torch.__version__ == '0.3.1':
            if cuda_device is None:
                self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
            elif isinstance(cuda_device, (int, list)):
                self._device = cuda_device
        else:
            if cuda_device is None:
                self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif cuda_device == 'cpu':
                self._device = torch.device("cpu")
            elif isinstance(cuda_device, int):
                self._device = torch.device(cuda_device)
            elif isinstance(cuda_device, list):
                self._device = cuda_device

        # init
        self.loss_fn = None
        self.optimizer = None
        if isinstance(model, str):
            self.model = None
            self.load(model)
        else:
            self.model = model
        self.compile(loss=loss_fn, optimizer=optimizer)
        self.save_folder = save_folder
        self._writer = SummaryWriter(save_folder)
        self._labels_and_predictions_store = OrderedDict()
        # counters
        self._iteration_count = 0  # train iter counter (.fit)
        self._eval_iteration_count = 0  # eval iter count (.evaluate)
        self._epoch_count = 0  # train epoch count
        # metrics and loss
        # last batch metrics and loss
        self._last_train_batch_log = OrderedDict()  # last train batch loss and metrics
        self._last_eval_batch_log = OrderedDict()  # last val batch loss and metrics
        # container for current epoch train loss and mterics
        self._last_train_epoch_log = []
        # epoch metrics and loss
        self._eval_metrics = OrderedDict()  # dict for (.evaluate) metrics values, use iteration as index
        self._epoch_metrics = OrderedDict()  # dict for (.fit) train metrics values, use epoch as index
        # store for callbacks
        self._step_callbacks = []
        self._epoch_begin_callbacks = []
        self._epoch_end_callbacks = []
        self._recurrent_callbacks = []
        self._true_stats = []
        self._pred_stats = []
    
    @staticmethod
    def as_torch(data, dtype='float32',
                 device=None, grad=False, async=False, **kwargs):
        """ Transform input data to torch Variable.

        Parameters
        ----------
        data : ndarray, torch Tensor or Variable
            input data.
        dtype : str
            data type.
        device : int or None
            gpu device. If None then cpu will be used.
        grad : bool
            whether data tensor require gradient computation.
        async : bool
            whether to enabl async mode.

        Returns
        -------
        Variable
        """
        if device.type == 'cpu':
            return _as_torch(data, dtype, None, grad, async, **kwargs)
        else:
            return _as_torch(data, dtype, device, grad, async, **kwargs)

