from pytorch_lightning import Trainer


def get_trainer(config):
    trainer = Trainer(
        # logger=...,
        gpus=config["gpus"],
        precision=config.get("precision", 32),
        # callbacks=...,
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        val_check_interval=config.get("val_check_interval", 1.0),
        limit_train_batches=config.get("train_percent", 1.0),
        limit_val_batches=config.get("val_percent", 1.0),
        progress_bar_refresh_rate=config.get("progress_bar_refresh_rate", 10),
        num_sanity_val_steps=config.get("sanity_steps", 5),
        log_every_n_steps=1,
    )
    return trainer
