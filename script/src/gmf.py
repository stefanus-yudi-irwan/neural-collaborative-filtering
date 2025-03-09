import torch
from loguru import logger
from .utils import detect_cuda
from .engine import Engine

class GeneralizeMatrixFactorizationEngine(Engine):
    """Class to make place GMF model object in CPU
       or GPU
    """
    def __init__(self, model_config : dict, dataset_config : dict):
        """GMF Model Engine Initialization
        Args:
            model_config (dict):
                configuration for GMF model
            dataset_config (dict):
                configuration from dataset
        """
        logger.info("BEGIN INITIALIZING MODEL GMF")
        # create model object
        self.model = GeneralizeMatrixFactorization(model_config, dataset_config)

        # place the model in CuDA if any
        if detect_cuda(device_id=model_config['DEVICE_ID']):
            self.model.cuda()

        # init engine class
        model_name = model_config['MODEL_NAME'].format(model_config['LATENT_DIM'],
                                                 model_config['NUM_NEGATIVE'],
                                                 model_config['BATCH_SIZE'],
                                                 model_config['L2_REGULARIZATION'],
                                                 dataset_config['MODEL_TYPE'])
        super().__init__(model_config = model_config,
                         dataset_config = dataset_config,
                         model_name = model_name)

        # print out self.model
        logger.info("FINISH INITIALIZING MODEL GMF")
        print(self.model)


class GeneralizeMatrixFactorization(torch.nn.Module):
    """Class to make Generalized Matrix Factorization
       object
    """
    def __init__(self, model_config : dict, dataset_config : dict):
        """Generalized Matrix Factorization Model
           initialization
        Args:
            model_config (dict):
                configuration for GMF model
            dataset_config (dict):
                configuration from dataset
        """
        logger.info("CREATE GMF MODEL LAYER")
        # initialization for torch.nn.Module
        super().__init__()

        # create layers embedding user and embedding item
        self.user_layer = torch.nn.Embedding(num_embeddings=dataset_config['NUM_USERS'], embedding_dim=model_config['LATENT_DIM'])     # size (num_user, n_laten)
        self.item_layer = torch.nn.Embedding(num_embeddings=dataset_config['NUM_ITEMS'], embedding_dim=model_config['LATENT_DIM'])     # size (num_item, n_laten)
        self.linear_layer = torch.nn.Linear(in_features=model_config['LATENT_DIM'], out_features=1, bias=False)
        self.sigmoid_layer = torch.nn.Sigmoid()
        
        # initialize layers weight
        self.weight_initialization()
        logger.info("GMF MODEL LAYERS CREATED")

    def forward(self,
                user_indices : torch.tensor,
                item_indices : torch.tensor) -> float:
        """Method for forward pass GMF
        Args:
            user_indices (torch.tensor):
                index of the user
            item_indices (torch.tensor):
                index of the item
        Returns:
            pred_proba:
                probability prediction of rating [0 1]
        """
        # slice user and item embedding
        user_embedding = self.user_layer(user_indices)
        item_embedding = self.item_layer(item_indices)

        # do element multiplication 
        element_product = torch.mul(user_embedding, item_embedding)

        # input to nn object
        logits = self.linear_layer(element_product)

        pred_proba_rating = self.sigmoid_layer(logits)

        return pred_proba_rating

    def weight_initialization(self):
        # initialize layers weights
        logger.info("INITIALIZE GMF LAYERS WEIGHT")
        torch.nn.init.kaiming_uniform_(self.user_layer.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.item_layer.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.linear_layer.weight.data, mode='fan_out', nonlinearity='linear')

    def load_gmf_pretrain_weights(self):
        pass