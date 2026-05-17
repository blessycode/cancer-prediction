import os
import tensorflow as tf

# =========================================================
# CALLBACKS
# =========================================================

def get_callbacks(save_path, append_log=False, monitor="val_accuracy", patience=8):

    os.makedirs(save_path, exist_ok=True)

    callbacks = [

        # ================================================
        # SAVE BEST MODEL
        # ================================================

        tf.keras.callbacks.ModelCheckpoint(

            filepath=os.path.join(
                save_path,
                "best_model.keras"
            ),

            monitor=monitor,

            save_best_only=True,

            mode="max" if "accuracy" in monitor else "min",

            verbose=1
        ),

        # ================================================
        # EARLY STOPPING
        # ================================================

        tf.keras.callbacks.EarlyStopping(

            monitor=monitor,

            patience=patience,

            restore_best_weights=True,

            mode="max" if "accuracy" in monitor else "min",

            verbose=1
        ),

        # ================================================
        # REDUCE LR
        # ================================================

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=max(2, patience // 3),

            min_lr=1e-7,

            verbose=1
        ),

        # ================================================
        # CSV LOGGER
        # ================================================

        tf.keras.callbacks.CSVLogger(

            os.path.join(
                save_path,
                "training_log.csv"
            ),

            append=append_log
        )

    ]

    return callbacks
