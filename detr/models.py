from typing import Any
from detr import DETR
import pytorch_lightning as L

class SuberbDetr(L.LightningModule):

    def __init__(self, n_classes: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = DETR(num_classes=n_classes + 1) # +1 for no object class

        self.save_hyperparameters()
    