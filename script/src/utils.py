import pandas as pd
import torch
import yaml
from loguru import logger

def create_optimizer(model : object , config : dict) -> object:
    """Function to craete optimizer based on config
    Args:
        network (object):
            model object (Generalized Matrix Factorization)
        params (dict):
            model configuration
    Returns:
        optimizer (object) : 
           optimizer object 
    """
    if config['OPTIMIZER'] == 'SGD':
        return torch.optim.SGD(model.parameters(), 
                               lr=config['SGD_LR'],
                               momentum=config['SGD_MOMENTUM'],
                               weight_decay=config['L2_REGULATIZATION'])
    elif config['OPTIMIZER'] == 'ADAM':
        return torch.optim.Adam(model.parameters(),
                                lr=config['ADAM_LR'],
                                weight_decay=config['L2_REGULARIZATION'])
    elif config['OPTIMIZER'] == 'RMSPROP':
        return torch.optim.RMSprop(model.parameters(),
                                   lr=config['RMSPROP_LR'],
                                   alpha=config['RMSPROP_ALPHA'],
                                   momentum=config['RMSPROP_MOMENTUM'])   
    else :
         logger.error("WRONG CONFIG OPTIMIZER, USE ONE OF THIS 'SDG', 'ADAM', OR 'RMSPROP'")

def create_loss_function(model_type : str) -> object:
    """Function to create loss function object
    Args:
        model_type (str): 
            model configuration
    Returns:
        loss_function (object):
            loss function for optimization 
    """
    if model_type == 'IMPLICIT':
        return torch.nn.BCELoss()
    elif model_type == 'EXPLICIT':
        return torch.nn.MSELoss()
    else:
        logger.error("WRONG CONFIG MODEL TYPE, USE 'IMPLICIT' OR 'EXPLICIT' INSTEAD")
    
def detect_cuda(device_id=0) -> bool:
    """Function to set device into CUDA GPU if any
    Args:
        device_id (int, optional):
            Device Cuda ID. Defaults to 0.
    Returns:
        CUDA status : 
            True : if CUDA GPU exists
            False : if CUDA GPU doesn't exists
    """
    logger.info("DETECTING CUDA DEVICE")
    if torch.cuda.is_available():
        logger.info("CUDA DEVICE DETECTED, SET DEVICE_ID INTO CUDA GPU")
        torch.cuda.set_device(device_id)
        return True
    else:
        logger.info("NO CUDA DEVICE DETECTED, USING CPU INSTEAD")
        return False
    
# Checkpoints
def save_checkpoint(model : object , model_dir : str):
    """Save model parameter to directory
    Args:
        model (object):
            The object of the model
        model_dir (str):
            Dictionary for saving the model
    """
    logger.info(f"SAVING MODEL STATEDICT AT : {model_dir}")
    torch.save(model.state_dict(), model_dir)
    logger.info("MODEL STATEDICT SAVED")

def resume_checkpoint(model : object, model_dir : str):
    """Load model parameter
    Args:
        model (object):
            model object GMF, MLP or NEUMF
        model_dir (str):
            Directory of source model object
    """
    logger.info(f"LOADING MODEL STATEDICT FROM : {model_dir}")
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)
    logger.info(f"MODEL STATEDICT LOADED")


def get_recommendation(user_id : int,
                       num_reco : int,
                       metadata : pd.DataFrame,
                       user_column_name : str,
                       item_column_name : str,
                       model_object : object,
                       data_generator_object : object) -> pd.DataFrame : 
    """Function to Predict Recommendation

    Args:
        user_id (int):
            User id
        num_reco (int):
            How much item for recommendation
        metadata (pd.DataFrame):
            Item dataframe metadata
        column_id (str):
            Id column in dataframe metadata
        model_object (object):
            model object for inference
        data_generator_object (object)):
            object data generator

    Returns:
        reco_df:
            dataframe containing recommendation for the user
    """
    # inference from model
    # get user model_id
    user_model_id = data_generator_object.mapped_dict['userid_to_usermodelid'][user_id]
    # predict proba from model
    negatives_item = list(data_generator_object.negatives_interaction['negative_items'][data_generator_object.negatives_interaction[user_column_name]==user_model_id].values[0])
    negatives_proba = model_object.model(torch.tensor(user_model_id).repeat(len(negatives_item)),
                                          torch.tensor(negatives_item))     # repeat to match user tensor and item tensor size

    # sort by probability descending
    # then retrieve the topk
    topk_movie_model_id = negatives_proba.t().argsort(descending=True).squeeze()[:num_reco].numpy()
    topk_proba_sorted_desc = negatives_proba[negatives_proba.t().argsort(descending=True)].squeeze()[:num_reco].detach().numpy()

    # get the movieID from movide Model Id
    movie_id = [data_generator_object.mapped_dict['itemmodelid_to_itemid'].get(x) for x in topk_movie_model_id]
    rec_df = pd.DataFrame({item_column_name : movie_id, "proba_score" : topk_proba_sorted_desc})

    # merge with metadata
    reco_df = metadata.merge(rec_df, how="right", on=item_column_name)

    return reco_df

def train_model(model : object, config : str, data_generator: object) : 
    """Function to train model object

    Args:
        model (object): 
             Model object to be trained
        config (str):
             Configuration for the model object
        data_generator (object):
             Data generator object for traning the model
        evaluation_data (list):
             Evaluation data for testing the model
    """
    # loop as every epoch
    batch_index = 0
    for epoch in range(config['NUM_EPOCH']):
        # generate the training data, sample negative interaction as per config
        # and create batch data as per batch size
        logger.info(f"CREATING DATA TRAIN LOADER | EPOCH {epoch} | NEGATIVE INTERACTION : {config['NUM_NEGATIVE']} | BATCH SIZE : {config['BATCH_SIZE']}")
        train_loader = data_generator.create_train_data(config['NUM_NEGATIVE'], config['BATCH_SIZE']) 
        logger.info(f"DATA LOADER DONE, START TRAINING MODEL EPOCH {epoch}")   # 1 batch contain 1024 data point

        # train the model using train_loader 
        batch_idx = model.train_an_epoch(train_loader, epoch_id=epoch, batch_index = batch_index)
        batch_index += batch_idx
        logger.info(f"EPOCH {epoch} DONE, START EVALUATING MODEL")

        # evaluate using evaluation data and 
        # calculate hit ratio and ndcg after and epoch finish
        hit_ratio, ndcg = model.evaluate(data_generator.evaluation_data, epoch_id=epoch)
        logger.info(f"MODEL EVALUATION DONE")

        # save model to directory defined in config
        model.save(config['MODEL_NAME'], epoch, hit_ratio, ndcg)

def load_config(file_path : str) -> dict:
    """Load config yaml files from path
    Args:
        file_path (str):
            yaml file path
    Returns:
        dict:
            dictionary configuration
    """
    with open(file_path, 'r') as stream:
        try: 
            logger.info("LOADING PROJECT CONFIGURATION")
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            logger.info(f"CONFIGURATION READ ERROR : {exc}")
            return None

