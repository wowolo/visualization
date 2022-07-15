# handle all the logging via one callback class
from pytorch_lightning import Callback


class LoggingCallback(Callback):
    def __init__(self, trainer):
        super().__init__()
        self.state = dict.fromkeys('x', 'y', 'output')
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.state.append(outputs)
        
    def on_train_epoch_end(self, trainer, pl_module):
        # access output using state
        all_outputs = self.state