from src.models.model_factory import get_model

model = get_model(

    model_name="mobilenetv2",

    input_shape=(224, 224, 3),

    num_classes=8

)

model.summary()