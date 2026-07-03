import importlib

REQUIRED_PACKAGES = [
    "torch",
    "pandas",
    "sklearn",
    "mlflow",
    "dvc",
]

for package in REQUIRED_PACKAGES:
    importlib.import_module(package)

print("Environment validated successfully")