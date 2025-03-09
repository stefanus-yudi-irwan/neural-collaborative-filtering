import torch
from loguru import logger
from .engine import Engine
from .utils import detect_cuda, resume_checkpoint
from .gmf import GeneralizeMatrixFactorization

class MultiLayerPerceptronEngine(Engine):
    """Class to make MLP model object in CPU
       or GPU
    """
    def __init__(self, model_config : dict, dataset_config : dict):
        """
        Args:
            model_config (dict):
                configuration for MLP model
            dataset_config (dict):
                configuration from dataset
        """
        logger.info("BEGIN INITIALIZING MODEl MLP")
        # create model object
        self.model = MultiLayerPerceptron(model_config, dataset_config)

        # place the model in CuDa if any
        if detect_cuda(device_id=model_config['DEVICE_ID']):
            self.model.cuda()
        
        # init engine class
        model_name = model_config['MODEL_NAME'].format(model_config['LATENT_DIM'],
                                                 model_config['NUM_NEGATIVE'],
                                                 model_config['BATCH_SIZE'],
                                                 model_config['L2_REGULARIZATION'],
                                                 "".join([str(x) for x in model_config['LAYERS']]),
                                                 dataset_config['MODEL_TYPE'])
        super().__init__(model_config = model_config,
                         dataset_config = dataset_config,
                         model_name =  model_name)

        # print out self.model
        print(self.model)

        # load weight from another model
        if model_config['PRETRAIN']:
            self.model.load_pretrain_weights(model_config, dataset_config)
        logger.info("FINISH INITIALIZING MODEL MLP")

class MultiLayerPerceptron(torch.nn.Module):
    """Class to make MultiLayerPerceptron Model
       object
    """
    def __init__(self, model_config : dict, dataset_config : dict):
        """Initialization for MLP Model
        Args:
            config (dict): 
                Model configuration
        """
        logger.info("CREATE MLP MODEL LAYERS")
        super().__init__()

        # intialize embedding user and embedding item
        self.user_layer = torch.nn.Embedding(num_embeddings=dataset_config['NUM_USERS'], embedding_dim=model_config['LATENT_DIM'])     # size (num_user, n_laten)
        self.item_layer = torch.nn.Embedding(num_embeddings=dataset_config['NUM_ITEMS'], embedding_dim=model_config['LATENT_DIM'])     # size (num_item, n_laten)

        self.fc_layers = []
        for _, (in_size, out_size) in enumerate(zip(model_config['LAYERS'][:-1], model_config['LAYERS'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.BatchNorm1d(num_features=out_size))
        self.fc_layers = torch.nn.Sequential(*self.fc_layers)

        self.linear_layer = torch.nn.Linear(in_features=model_config['LAYERS'][-1], out_features=1, bias=False)
        self.sigmoid_layer = torch.nn.Sigmoid()

        # initialize weight 
        self.weight_initialization()
        logger.info("MLP MODEL LAYERS CREATED")

    def forward(self,
                user_indices : torch.tensor,
                item_indices : torch.tensor):
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

        # create concatenated vector user and item
        vector = torch.cat([user_embedding, item_embedding], dim=-1) # the concat latent vector

        # forward pass to multi layer perceptron layer
        vector = self.fc_layers(vector) 

        # input to nn object
        logits = self.linear_layer(vector)

        pred_proba_rating = self.sigmoid_layer(logits)

        return pred_proba_rating

    def weight_initialization(self):
        # initialize layers weights
        logger.info("INITIALIZE MLP LAYERS WEIGHT")
        torch.nn.init.kaiming_uniform_(self.user_layer.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.item_layer.weight.data, mode='fan_in', nonlinearity='linear')

        # Apply Kaiming Initialization to fc layers
        for layer in self.fc_layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                # Optionally, initialize biases to zeros or other suitable values
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias.data)

        # initialize linear layer
        torch.nn.init.kaiming_uniform_(self.linear_layer.weight.data, mode='fan_out', nonlinearity='linear')

    def load_gmf_pretrain_weights(self,
                                model_config : dict,
                                dataset_config : dict):
        """Method to initialize embedding layer weight 
           of MLP from GMF model
        """
        logger.info("LOAD USER AND ITEM WEIGHT FROM GMF MODEL")
        gmf_model = GeneralizeMatrixFactorization(model_config = model_config,
                                                  dataset_config = dataset_config)
        if detect_cuda(device_id=model_config['DEVICE_ID']):
            gmf_model.cuda()

        resume_checkpoint(gmf_model, model_dir=model_config['PRETRAIN_GMF'], device_id=model_config['DEVICE_ID'])
        self.user_layer.weight.data = gmf_model.user_layer.weight.data
        self.item_layer.weight.data = gmf_model.item_layer.weight.data