import pandas as pd
from src.utils import load_config
from src.data import NCFDataset

# load configuration
config_path = "config/config.yaml"
config = load_config(file_path=config_path)

# load raw and basic preprocess raw data
interaction_data_dir = "../data/raw/ml-1m/ratings.dat"
movie_metadata_dir = "../data/raw/ml-1m/movies.dat"
interaction_data = pd.read_csv(interaction_data_dir, sep="::", header=None, names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')
movie_metadata = pd.read_csv(movie_metadata_dir, sep="::", header=None, names=["MovieID", "Title", "Genres"], engine='python', encoding='latin-1')
interaction_data.dropna(inplace=True)
interaction_data.drop_duplicates(inplace=True)
movie_metadata.dropna(inplace=True)
movie_metadata.drop_duplicates(inplace=True)

# process raw data
NCFDatasetObj = NCFDataset(config_data = config["DatasetConfig"])
NCFDatasetObj.process_raw_data(raw_data=interaction_data)
NCFDatasetObj.save_dataset()