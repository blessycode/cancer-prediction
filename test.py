from src.data_loader.dataset import load_datasets

train_ds, val_ds, test_ds = load_datasets(
    "data/processed/224x224",
    (224, 224)
)

print(train_ds)
print(val_ds)
print(test_ds)