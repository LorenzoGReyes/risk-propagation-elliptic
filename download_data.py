import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    os.makedirs("data", exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files(
            "ellipticco/elliptic-data-set",
            path="data",
            unzip=True
        )
        print("Data downloaded")
    except Exception as e:
        print(f"{e}")
        

if __name__ == "__main__":
    main()