import tensorflow as tf
import os
from src.data_loader.augmentation import train_augmentation

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# LOAD DATASET
# =========================================================

def load_datasets(data_dir, image_size):

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # =====================================================
    # TRAIN DATASET
    # =====================================================

    train_dataset = tf.keras.utils.image_dataset_from_directory(

        train_dir,

        image_size=image_size,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    # =====================================================
    # VALIDATION DATASET
    # =====================================================

    val_dataset = tf.keras.utils.image_dataset_from_directory(

        val_dir,

        image_size=image_size,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    # =====================================================
    # TEST DATASET
    # =====================================================

    test_dataset = tf.keras.utils.image_dataset_from_directory(

        test_dir,

        image_size=image_size,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    val_dataset = val_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    # =====================================================
    # AUGMENT TRAIN ONLY
    # =====================================================

    train_dataset = train_dataset.map(
        lambda x, y: (train_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
