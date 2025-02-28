from thesis_utils.file_utils import deterministic_run
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L


def setup_trainer(name: str, seed: int = 0, save_dir="/home/tha/thesis_runs/regressor"):
    deterministic_run(seed=seed)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        filename="{step:04d}-{val_loss:.3e}" + f"-seed={seed}",
    )

    # Add your callback to the callbacks list
    logger = TensorBoardLogger(save_dir, name=name)

    MAX_STEPS = 120_000  # Assuming Batch Size 32
    val_check_every_steps = int(MAX_STEPS / 20)
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        check_val_every_n_epoch=None,
        val_check_interval=val_check_every_steps,
        log_every_n_steps=5,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    return trainer
