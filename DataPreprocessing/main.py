from DataPreprocessing.extract_crops import extract_crops
def main():
    extract_crops(input_path="./DataSet/MTMC_Tracking_2025/train/Warehouse_000",output_path="./DataSet/MTMC_Tracking_2025_Preprocessed/train/Warehouse_000")

if __name__ == "__main__":
    main()

    
    # videos = [
    #     folder for folder in os.listdir(videos_path)
    #     if os.path.isdir(os.path.join(videos_path, folder))
    # ]