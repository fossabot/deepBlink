"""SpotsModel class."""

import functools

import numpy as np
import tensorflow as tf

from ..augment import augment_batch_baseline
from ..losses import f1_l2_combined_loss
from ..losses import f1_score
from ..losses import l2_norm
from ._models import Model

DEFAULT_TRAIN_ARGS = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 3e-04,
}
DEFAULT_NETWORK_ARGS = {"n_classes": 3}
DEFAULT_LOSS = tf.keras.losses.binary_crossentropy
DEFAULT_OPTIMIZER = tf.keras.optimizers.Adam(DEFAULT_TRAIN_ARGS["learning_rate"])


class SpotsModel(Model):
    """Class to predict spot localization; see base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_augment_fn = functools.partial(
            augment_batch_baseline,
            flip_=self.dataset_args["flip"],
            illuminate_=self.dataset_args["illuminate"],
            gaussian_noise_=self.dataset_args["gaussian_noise"],
            rotate_=self.dataset_args["rotate"],
            translate_=self.dataset_args["translate"],
            cell_size=self.dataset_args["cell_size"],
        )

    @property
    def metrics(self) -> list:
        """Metrics used in the training."""
        return [
            "accuracy",
            "mse",
            f1_score,
            l2_norm,
            f1_l2_combined_loss,
        ]

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single input image."""
        return self.network.predict(image[None, ..., None], batch_size=1).squeeze()