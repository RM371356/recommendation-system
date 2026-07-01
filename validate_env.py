import importlib

# Este script é responsável por validar o ambiente de desenvolvimento, verificando se todas as dependências necessárias estão instaladas. 
# Ele tenta importar cada pacote listado em REQUIRED_PACKAGES.
REQUIRED_PACKAGES = [
    "torch",
    "pandas",
    "sklearn",
    "mlflow",
    "dvc",
]

# Tenta importar cada pacote listado em REQUIRED_PACKAGES para verificar se estão instalados no ambiente. 
# Se algum pacote não estiver instalado, uma ImportError será levantada, indicando que o ambiente não está configurado corretamente.
for package in REQUIRED_PACKAGES:
    importlib.import_module(package)

print("Environment validated successfully")