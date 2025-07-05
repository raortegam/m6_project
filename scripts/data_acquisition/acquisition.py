
import kaggle
kaggle.api.dataset_download_files('blastchar/telco-customer-churn', 
                                 path='./src/database', 
                                 unzip=True)
print("Archivo descargado exitosamente")