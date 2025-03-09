import uvicorn
import pandas as pd
from loguru import logger
from fastapi import FastAPI, Response
from src.utils import get_recommendation, resume_checkpoint, load_config
from src.data import NCFDataset
from src.gmf import GeneralizeMatrixFactorizationEngine
from src.mlp import MultiLayerPerceptronEngine
from src.neumf import NeuMatrixFactorizationEngine

# create FAST API instance
app = FastAPI()

# create route
HEALTH = '/ncf/health'
PREDICT_GMF = '/ncf/gmf/predict/{user_id}/{num_reco}'
PREDICT_MLP = '/ncf/mlp/predict/{user_id}/{num_reco}'
PREDICT_NEUMF = '/ncf/neumf/predict/{user_id}/{num_reco}'

# load configuration
config_path = "config/config.yaml"
config = load_config(file_path=config_path)

# load metadata
movie_metadata_dir = "../data/raw/ml-1m/movies.dat"
movie_metadata = pd.read_csv(movie_metadata_dir, sep="::", header=None, names=["MovieID", "Title", "Genres"], engine='python', encoding='latin-1')

# create object dataset
NCFDatasetObj = NCFDataset(config_data = config['DatasetConfig'])
NCFDatasetObj.load_dataset()
DatasetModelConfig = {
    "NUM_USERS" : len(NCFDatasetObj.user_pool),
    "NUM_ITEMS" : len(NCFDatasetObj.item_pool),
    "MODEL_TYPE" : config['DatasetConfig']['MODEL_TYPE'],}

# list pretrain model
gmf_pretrain_model = "../notebook/checkpoints/gmf/GMF_FACTOR-8_NEG-4_BS-1024_REGL2-1e-06_MODELTYPE-IMPLICIT_EPOCH0_HR0.1118_NDCG0.0530.model"
mlp_pretrain_model = "../notebook/checkpoints/mlp/MLP_FACTOR-8_NEG-4_BS-1024_REGL2-1e-06_LAYERS-166432168_MODELTYPE-IMPLICIT_EPOCH0_HR0.4442_NDCG0.2473.model"
neumf_pretrain_model = "../notebook/checkpoints/neumf/NEUMF_FACTORGMF-8_FACTORMLP-8_NEG-4_BS-1024_REGL2-1e-06_LAYERS-1632168_MODELTYPE-IMPLICIT_EPOCH0_HR0.4508_NDCG0.2523.model"

# create GMF model object
GMFEngine = GeneralizeMatrixFactorizationEngine(model_config = config['GMFConfig'], dataset_config = DatasetModelConfig)
resume_checkpoint(model = GMFEngine.model, model_dir = gmf_pretrain_model)

# create MLP model object
MLPEngine = MultiLayerPerceptronEngine(model_config = config['MLPConfig'], dataset_config = DatasetModelConfig)
resume_checkpoint(model = MLPEngine.model, model_dir = mlp_pretrain_model)

# create NeuMLP model object
NEUMFEngine = NeuMatrixFactorizationEngine(model_config = config['NEUMFConfig'], dataset_config = DatasetModelConfig)
resume_checkpoint(model = NEUMFEngine.model, model_dir = neumf_pretrain_model)

@app.get(HEALTH)
def check_health(response: Response):
    """Check API Health
    Args:
        dict: health status of the api
    """
    logger.info("CHECKING API HEALTH")
    response.headers["Content-Type"] = "application/json"
    api_status = {
        "apiVersion":1.0,
        "apiStatus":"OK"
    }
    return api_status

@app.get(PREDICT_GMF)
async def recommend_gmf(user_id: int, num_reco: int):
    """Recommendation Using GMF Model
    Args:
        user_id (int):
            user_id registered
        num_reco (int):
            number of film will be recommended
    Returns:
        dict:
            recommendation payload
    """
    # call function recommendation
    reco_df_gmf = get_recommendation(user_id = user_id,
                                num_reco = num_reco,
                                metadata = movie_metadata,
                                user_column_name = "UserID",
                                item_column_name = "MovieID",
                                model_object = GMFEngine,
                                data_generator_object = NCFDatasetObj)
    # format into json response
    reco_res = reco_df_gmf.to_dict(orient='records')

    return { "user_id" : user_id,
            "num_reco" : num_reco,
            "film_reco" : reco_res}

@app.get(PREDICT_MLP)
async def recommend_mlp(user_id: int, num_reco: int):
    """Recommendation Using MLP Model
    Args:
        user_id (int):
            user_id registered
        num_reco (int):
            number of film will be recommended
    Returns:
        dict:
            recommendation payload
    """
    # call function recommendation
    reco_df_mlp = get_recommendation(user_id = user_id,
                                num_reco = num_reco,
                                metadata = movie_metadata,
                                user_column_name = "UserID",
                                item_column_name = "MovieID",
                                model_object = MLPEngine,
                                data_generator_object = NCFDatasetObj)
    # format into json response
    reco_res = reco_df_mlp.to_dict(orient='records')

    return { "user_id" : user_id,
            "num_reco" : num_reco,
            "film_reco" : reco_res}

@app.get(PREDICT_NEUMF)
# get input from path parameter
async def recommend_neumf(user_id: int, num_reco: int):
    """Recommendation Using NEUMF Model
    Args:
        user_id (int):
            user_id registered
        num_reco (int):
            number of film will be recommended
    Returns:
        dict:
            recommendation payload
    """
    # call function recommendation
    reco_df_neumf = get_recommendation(user_id = user_id,
                                num_reco = num_reco,
                                metadata = movie_metadata,
                                user_column_name = "UserID",
                                item_column_name = "MovieID",
                                model_object = NEUMFEngine,
                                data_generator_object = NCFDatasetObj)
    # format into json response
    reco_res = reco_df_neumf.to_dict(orient='records')

    return { "user_id" : user_id,
            "num_reco" : num_reco,
            "film_reco" : reco_res}

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)