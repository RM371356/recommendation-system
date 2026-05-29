from importlib.metadata import version

REQUIRED_PACKAGES = [
"pandas",
"numpy",
"scikit-learn",
"torch",
"mlflow",
"dvc",
]

def validate_packages() -> None:
    """Validate installed packages."""
    for package in REQUIRED_PACKAGES:
        package_version = version(package)
        print(f"{package}: {package_version}")

if __name__ == "__main__":
    validate_packages()