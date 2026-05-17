import tensorflow as tf

# =========================================================
# TRAIN AUGMENTATION
# =========================================================

train_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.08),

    tf.keras.layers.RandomTranslation(0.08, 0.08),

    tf.keras.layers.RandomZoom(0.12),

    tf.keras.layers.RandomContrast(0.18),

    tf.keras.layers.GaussianNoise(0.015)

])
