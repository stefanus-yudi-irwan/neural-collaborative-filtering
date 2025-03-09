import torch
from loguru import logger
from .gmf import GeneralizeMatrixFactorization
from .mlp import MultiLayerPerceptron
from .engine import Engine
from .utils import detect_cuda, resume_checkpoint


class NeuMatrixFactorizationEngine(Engine):
    """Class to make MLP model object in CPU
    or GPU
    """
    def __init__(self,
                 model_config : dict,
                 dataset_config : dict):
        """NeuMF Model Engine Initialization
        Args:
            model_config (dict):
                configuration for NeuMF model
            dataset_config (dict):
                configuration from dataset
        """
        logger.info("BEGIN INITIALIZING MODEL NEUMF")
        # create model object
        self.model = NeuMatrixFactorization(model_config,
                                            dataset_config)

        # place the model in CuDA if any
        if detect_cuda(device_id=model_config['DEVICE_ID']):
            self.model.cuda()

        # init engine class
        model_name = model_config['MODEL_NAME'].format(model_config['LATENT_DIM_GMF'],
                                                 model_config['LATENT_DIM_MLP'],
                                                 model_config['NUM_NEGATIVE'],
                                                 model_config['BATCH_SIZE'],
                                                 model_config['L2_REGULARIZATION'],
                                                 "".join([str(x) for x in model_config['LAYERS']]),
                                                 dataset_config['MODEL_TYPE'])
        super().__init__(model_config = model_config,
                         dataset_config = dataset_config, 
                         model_name = model_name)

        # print out self.model
        print(self.model)

        # load weight from others model
        if model_config['PRETRAIN']:
            self.model.load_pretrain_weights(model_config, dataset_config)
        logger.info("FINISH INITIALIZING MODEL NEUMF")

class NeuMatrixFactorization(torch.nn.Module):
    def __init__(self,
                 model_config : dict,
                 dataset_config : dict):
        """Initialization for Neu Matrix Factorization 
           Model

        Args:
            config (dict): 
                Model configuration
        """
        logger.info("CREATE NEUMF MODEL LAYERS")
        super().__init__()
        # intialize embedding user and embedding item
        self.user_layer_mlp = torch.nn.Embedding(num_embeddings=dataset_config['NUM_USERS'], embedding_dim=model_config['LATENT_DIM_MLP'])     # size (num_user, n_laten_mlp)
        self.item_layer_mlp = torch.nn.Embedding(num_embeddings=dataset_config['NUM_ITEMS'], embedding_dim=model_config['LATENT_DIM_MLP'])     # size (num_item, n_laten_mlp)
        self.user_layer_mf = torch.nn.Embedding(num_embeddings=dataset_config['NUM_USERS'], embedding_dim=model_config['LATENT_DIM_GMF'])     # size (num_user, n_laten_mf)
        self.item_layer_mf = torch.nn.Embedding(num_embeddings=dataset_config['NUM_ITEMS'], embedding_dim=model_config['LATENT_DIM_GMF'])     # size (num_item, n_laten_mf)

        self.fc_layers = []
        for _, (in_size, out_size) in enumerate(zip(model_config['LAYERS'][:-1], model_config['LAYERS'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.BatchNorm1d(num_features=out_size))
        self.fc_layers = torch.nn.Sequential(*self.fc_layers)

        self.linear_layer = torch.nn.Linear(in_features=model_config['LAYERS'][-1] + model_config['LATENT_DIM_GMF'], out_features=1, bias=False)
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
        user_embedding_mlp = self.user_layer_mlp(user_indices)
        item_embedding_mlp = self.item_layer_mlp(item_indices)
        user_embedding_mf = self.user_layer_mf(user_indices)
        item_embedding_mf = self.item_layer_mf(item_indices)

        # create concatenated mlp vector user and item
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1) # the concat latent vector

        # process mlp vector through NN
        mlp_vector = self.fc_layers(mlp_vector)

        # process mf_vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # concat 2 vectors
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        # shrink to dim = 1
        logits = self.linear_layer(vector)

        pred_proba_rating = self.sigmoid_layer(logits)

        return pred_proba_rating
    
    def weight_initialization(self):
        # initialize embedding weght
        logger.info("INITIALIZE NEUMF LAYERS WEIGHT")
        torch.nn.init.kaiming_uniform_(self.user_layer_mlp.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.item_layer_mlp.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.user_layer_mf.weight.data, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.item_layer_mf.weight.data, mode='fan_in', nonlinearity='linear')

        # Apply Kaiming Initialization to fc layers
        for layer in self.fc_layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                # Optionally, initialize biases to zeros or other suitable values
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias.data)

        # initialize linear layer
        torch.nn.init.kaiming_uniform_(self.linear_layer.weight.data, mode='fan_out', nonlinearity='linear')

    def load_pretrain_weights(self,
                              model_config : dict,
                              dataset_config : dict):
        """Method to load weights from GMF and MLP
        """
        logger.info("LOAD USER AND ITEM WEIGHT FROM GMF MODEL")
        config = model_config
        # configure MLP Model
        config['LATENT_DIM'] = config['LATENT_DIM_MLP']
        mlp_model = MultiLayerPerceptron(model_config = config,
                                         dataset_config = dataset_config)
        if detect_cuda(device_id=config['DEVICE_ID']):
            mlp_model.cuda()
        resume_checkpoint(mlp_model, model_dir=config['PRETRAIN_MLP'], device_id=config['DEVICE_ID'])
        self.user_layer_mlp.weight.data = mlp_model.user_layer.weight.data
        self.item_layer_mlp.weight.data = mlp_model.item_layer.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        # configure MF Model
        config['LATENT_DIM'] = config['LATENT_DIM_GMF']
        gmf_model = GeneralizeMatrixFactorization(model_config = config,
                                                  dataset_config = dataset_config)
        if detect_cuda(device_id=config['DEVICE_ID']):
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['PRETRAIN_GMF'], device_id=config['DEVICE_ID'])
        self.user_layer_mf.weight.data = gmf_model.user_layer.weight.data
        self.item_layer_mf.weight.data = gmf_model.user_layer.weight.data

        self.linear_layer.weight.data = 0.5 * torch.cat([mlp_model.linear_layer.weight.data, gmf_model.linear_layer.weight.data], dim=-1)
        self.linear_layer.bias.data = 0.5 * (mlp_model.linear_layer.bias.data + gmf_model.linear_layer.bias.data)