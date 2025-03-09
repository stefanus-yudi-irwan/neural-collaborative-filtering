import torch
from loguru import logger
from tensorboardX import SummaryWriter
from .metrics import MetronAtk
from .utils import save_checkpoint, create_optimizer, create_loss_function

class Engine(object):
    """Class for handling training and evaluation
       process
    """
    def __init__(self,
                 model_config : dict,
                 dataset_config : dict,
                 model_name : str):
        """Training and Evaluation Engine initialization
        Args:
            model_config (dict):
                Model configuration
            dataset_config (dict):
                configuration from dataset
        """
        # copy model configuration
        self.model_config = model_config.copy()
        self.dataset_config = dataset_config.copy()
        self.model_name = model_name     

        # create MetronAtk Object
        self._metron = MetronAtk(top_k = self.model_config['TOP_K'])

        # create SummaryWriter Object
        self._writer = SummaryWriter(log_dir=f"runs/{self.model_name}")
        self._writer.add_text('MODEL-CONFIG', str(self.model_config), 0)

        # create optimizer object
        self.optimizer = create_optimizer(model = self.model, config = self.model_config)

        # create loss object
        self.loss_function = create_loss_function(model_type=self.dataset_config["MODEL_TYPE"])
    
    def train_single_batch(self,
                           users : torch.LongTensor,
                           items : torch.LongTensor,
                           interactions : torch.FloatTensor) -> float:
        """Method for training model in one batch data
        Args:
            users (torch.LongTensor):
                one batch data from the user data
            items (torch.LongTensor):
                one batch data from the item data
            interactions (torch.FloatTensor):
                one batch data from the interactions data
        Returns:
            loss (float):
                total loss from one batch data 
        """
        # assert if the object has model attribute
        assert hasattr(self, 'model'), 'Create self.model first!'

        # if cuda is used move the tensor to cuda
        if self.model_config['USE_CUDA']:
            users, items, interactions = users.cuda(), items.cuda(), interactions.cuda()

        # zero grad the optimizer   
        self.optimizer.zero_grad()
        pred_proba_rating = self.model(users, items)     # forward pass
        loss = self.loss_function(pred_proba_rating.view(-1), interactions)     # calculate loss
        loss.backward()     # backpropagation
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader : object, epoch_id : int, batch_index : int):
        """Method For training model in one epoch
        Args:
            train_loader (object):
                DataLoader object 
            epoch_id (int):
                Identifier for the epoch
        """
        # assert if the object has model attribute
        assert hasattr(self, 'model'), 'Create self.model first!'

        self.model.train()     # Turn on the train mode
        total_loss = 0     # initialize total loss
        batch_idx = batch_index
        # enumerate dataLoader => id, data
        for batch_id, batch in enumerate(train_loader):

            # assert every data type user, item, interaction
            assert isinstance(batch[0], torch.LongTensor)     
            assert isinstance(batch[1], torch.LongTensor)
            assert isinstance(batch[2], torch.FloatTensor)

            users, items, interactions = batch[0], batch[1], batch[2]
            interactions = interactions.float()
            loss = self.train_single_batch(users, items, interactions)
            self._writer.add_scalar('LOSS/PER_BATCH', loss, batch_idx)
            batch_idx += 1
            if batch_id % 1000 == 0:
                logger.info(f"[TRAINING EPOCH {epoch_id}] BATCH {batch_id}, LOSS {loss}")
            total_loss += loss
        self._writer.add_scalar('LOSS/PER_EPOCH', total_loss, epoch_id)

        return batch_idx

    def evaluate(self, evaluation_data : list, epoch_id : int) -> (float, float):
        """Method to evaluate model using test data

        Args:
            evaluation_data (list):
                test data tensor value inside a list
            epoch_id (int):
                identifier for epoch process

        Returns:
            hit_ratio, ndcg (float, float):
                optimization metrics for recommender system
        """
        # assert if the object has model attribute
        assert hasattr(self, 'model'), 'Create self.model first!'

        self.model.eval()     # turn on evaluation mode
        with torch.no_grad():
            # get the tensor evaluation -data
            positive_users, positive_items = evaluation_data[0], evaluation_data[1]
            negative_users, negative_items = evaluation_data[2], evaluation_data[3]

            # if using cuda
            if self.model_config['USE_CUDA']:
                positive_users = positive_users.cuda()
                positive_items = positive_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            predict_positive_interaction = self.model(positive_users, positive_items)
            predict_negative_interaction = self.model(negative_users, negative_items)

            # move back to cpu
            if self.model_config['USE_CUDA']:
                positive_users = positive_users.cpu()
                positive_items = positive_items.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                predict_positive_interaction = predict_positive_interaction.cpu()
                predict_negative_interaction = predict_negative_interaction.cpu()
            
            self._metron.subjects = [positive_users.data.view(-1).tolist(),
                                     positive_items.data.view(-1).tolist(),
                                     predict_positive_interaction.data.view(-1).tolist(),
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     predict_negative_interaction.data.view(-1).tolist()]
            
            hit_ratio, ndcg = self._metron.calculate_hit_ratio(), self._metron.calculate_ndcg()
            self._writer.add_scalar('PERFORMANCE/HIT_RATIO', hit_ratio, epoch_id)
            self._writer.add_scalar('PERFORMANCE/NDCG', ndcg, epoch_id)
            logger.info(f"[EVALUATING EPOCH {epoch_id}] HIT_RATIO = {hit_ratio:.4f}, NDCG = {ndcg:.4f}")
            return hit_ratio, ndcg
        
    def save(self, alias : str, epoch_id : int, hit_ratio : float, ndcg : float):
        """Method to save model to directory

        Args:
            alias (str):
                the name of the model
            epoch_id (int):
                the identifier of the epochs
            hit_ratio (float):
                Hit ratio of the model
            ndcg (float):
                NDCG of the model
        """
        # assert if the object has model attribute
        assert hasattr(self, 'model'), 'Create self.model first!'
        model_dir = self.model_config['MODEL_DIR'].format(self.model_name, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
