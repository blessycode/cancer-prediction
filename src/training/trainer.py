import tensorflow as tf

from src.training.callbacks import get_callbacks

# =========================================================
# TRAIN MODEL
# =========================================================

def _get_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.layers:
            return layer
    return None


def _prepare_backbone_for_fine_tuning(base_model, unfreeze_last_n):
    base_model.trainable = True

    trainable_start = max(len(base_model.layers) - unfreeze_last_n, 0)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= trainable_start

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def train_model(

    model,

    train_dataset,

    val_dataset,

    save_path,

    epochs=30,

    learning_rate=0.001,
    fine_tune=False,
    fine_tune_epochs=10,
    fine_tune_learning_rate=1e-5,
    unfreeze_last_n=30,
    class_weight=None,
    monitor="val_accuracy"

):

    # =====================================================
    # COMPILE MODEL
    # =====================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ]

    )

    # =====================================================
    # CALLBACKS
    # =====================================================

    callbacks = get_callbacks(save_path, monitor=monitor, patience=8)

    # =====================================================
    # TRAINING
    # =====================================================

    history = model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=epochs,

        callbacks=callbacks,

        class_weight=class_weight

    )

    if not fine_tune:
        return history

    base_model = _get_backbone(model)

    if base_model is None:
        return history

    _prepare_backbone_for_fine_tuning(base_model, unfreeze_last_n)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    fine_tune_callbacks = get_callbacks(save_path, append_log=True, monitor=monitor, patience=10)

    fine_tune_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs + fine_tune_epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=fine_tune_callbacks,
        class_weight=class_weight,
    )

    for key, values in fine_tune_history.history.items():
        history.history.setdefault(key, []).extend(values)

    return history
