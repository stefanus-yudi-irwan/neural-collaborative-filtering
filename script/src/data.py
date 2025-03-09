import random
import pandas as pd
import pickle
from copy import deepcopy
from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset

class NCFDataset(object):
    """Dataset provider for Neural Colaborative Filtering"""

    def __init__(self,
                 config_data : dict) -> None:
        """Initialization DataGenerator Object
        Args:
            config_data (dict):
                configuration for DataGenerator object 
        """
        logger.info("INITIALIZING NCF DATASET")
        random.seed(666)
        # save dataset configuration
        self.model_type = config_data['MODEL_TYPE']
        self.user_name = config_data['USER_COLUMN_NAME']
        self.item_name = config_data['ITEM_COLUMN_NAME']
        self.interaction_name = config_data['INTERACTION_COLUMN_NAME']
        self.timestamp_name = config_data['TIMESTAMP_COLUMN_NAME']
        self.negative_interaction_sample = config_data['NEGATIVE_INTERACTION_SAMPLE']
        self.dataset_dir = config_data['DATASET_DIR']

    def process_raw_data(self, raw_data : pd.DataFrame) -> None:
        """Process raw data into NCF dataset for training and
           evaluation
        Args:
            raw_data (pd.DataFrame):
                raw data which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        Raises:
            KeyError:
                raise error if model type config not listed
        """
        logger.info("BEGIN PROCESSING RAW DATA")
        # create mapped data
        self.mapped_data, self.mapped_dict = self._create_mapped_data(raw_data)
        
        # explicit feedback use _normalize and implicit use _binarize
        if self.model_type == 'EXPLICIT':
            self.mapped_data = self._scale_interaction()
        elif self.model_type == 'IMPLICIT':
            self.mapped_data = self._binarize_interaction()
        else:
            logger.error("DEFINE THE RIGHT CONFIG MODEL TYPE 'IMPLICIT' OR 'EXPLICIT'")

        # calculate data sparsity
        self._calculate_data_sparsity()
        
        # create negative item samples for NCF learning
        self.negatives_interaction = self._create_negative_interaction()
        
        # split train and test data
        self.train_data, self.test_data = self._split_train_test()

        # create evaluation data
        self.evaluation_data = self.create_evaluation_data()

    def save_dataset(self):
        logger.info(f"SAVE PROCESSED DATASET TO : {self.dataset_dir}")
        pickle.dump((self.mapped_data, self.mapped_dict, self.negatives_interaction, self.train_data, self.test_data, self.evaluation_data), 
                    open(self.dataset_dir, "wb"))
        logger.info("PROCESSED DATASET SAVED")

    def load_dataset(self):
        logger.info(f"LOAD DATASET FROM : {self.dataset_dir}")
        dataset = pickle.load(open(self.dataset_dir, "rb"))
        self.mapped_data = dataset[0]
        self.mapped_dict = dataset[1]
        self.negatives_interaction = dataset[2]
        self.train_data = dataset[3]
        self.test_data = dataset[4]
        self.evaluation_data = dataset[5]
        logger.info("DATASET LOADED")
        self._calculate_data_sparsity()

    def _create_mapped_data(self,
                            raw_data : pd.DataFrame) -> (pd.DataFrame, dict):
        """Methcod to map user_id and item_id to interger
        Args:
            data (pd.DataFrame): raw_data user and item interaction
        Returns:
            mapped_data: mapped user_id and item_id to id for modeling
            mapped_dict: dictionary for mapping real id to model id
        """
        logger.info("BEGIN CREATING MAPPED DICT AND MAPPED DATA")
        logger.info(f"THE NUMBER OF UNIQUE USE IS {raw_data[self.user_name].nunique()}, AND UNIQUE ITEM ID IS {raw_data[self.item_name].nunique()}")

        # create mapping value of user and item
        userid_to_usermodelid = {user : idx for idx, user in enumerate(raw_data[self.user_name].unique())}
        usermodelid_to_userid = {idx : user for idx, user in enumerate(raw_data[self.user_name].unique())}
        itemid_to_itemmodelid = {item : idx for idx, item in enumerate(raw_data[self.item_name].unique())}
        itemmodelid_to_itemid = {idx : item for idx, item in enumerate(raw_data[self.item_name].unique())}

        # sanity check mapping
        assert len(userid_to_usermodelid) == raw_data[self.user_name].nunique(), "user map value doesn't fit with unique user_id"
        assert len(itemid_to_itemmodelid) == raw_data[self.item_name].nunique(), "item map value doesn't fit with unique item_id"

        # create the mapped data frame
        mapped_data = raw_data.copy()
        mapped_data[self.user_name] = mapped_data[self.user_name].apply(lambda x: userid_to_usermodelid[x])
        mapped_data[self.item_name] = mapped_data[self.item_name].apply(lambda x: itemid_to_itemmodelid[x])

        # create map dictionary
        mapped_dict = {"userid_to_usermodelid" :  userid_to_usermodelid,
                    "usermodelid_to_userid" : usermodelid_to_userid,
                    "itemid_to_itemmodelid" : itemid_to_itemmodelid,
                    "itemmodelid_to_itemid" : itemmodelid_to_itemid}
        
        logger.info("FINISH CREATING MAPPED DICT AND MAPPED DATA")
        return mapped_data, mapped_dict

    def _scale_interaction(self) -> pd.DataFrame:
        """Scale mapped interaction data into
        [0 1] for explicit interaction
        Returns:
            normalized_data (pd.DataFrame): 
                data frame with normalized interaction data        
        """
        logger.info("START SCALING INTERACTION DATA")
        scaled_data = deepcopy(self.mapped_data)
        max_rating = scaled_data[self.interaction_name].max()
        scaled_data[self.interaction_name] = scaled_data[self.interaction_name] * 1.0 / max_rating
        logger.info("FINISH SCALING INTERACTION DATA")
        return scaled_data
    
    def _binarize_interaction(self) -> pd.DataFrame:
        """Binarized mapped interaction data into
        [0, 1] for implicit interaction
        Returns:
            normalized_data (pd.DataFrame): 
                data frame with binarized interaction data        
        """
        logger.info("START BINARIZING INTERACTION DATA")
        binarized_data = deepcopy(self.mapped_data)
        binarized_data.loc[binarized_data[self.interaction_name] > 0,self.interaction_name] = 1.0
        logger.info("FINISH BINARIZING INTERACTION DATA")
        return binarized_data
    
    def _calculate_data_sparsity(self) -> None:
        """Calculate Density and Sparsity of the 
           Data
        """
        logger.info("CALCULATE DATA SPARSITY")
        # save unique user and item
        self.user_pool = set(self.mapped_data[self.user_name].unique())
        self.item_pool = set(self.mapped_data[self.item_name].unique())

        density = len(self.mapped_data) / (len(self.user_pool)*len(self.item_pool))
        sparsity = 1 - density

        logger.info(f"THE DATA DENSITY IS {density:.4f} AND THE DATA SPARSITY IS {sparsity:.4f}")
    
    def _create_negative_interaction(self) -> pd.DataFrame:
        """Return all negative items from a user & sampled negative items
        Returns:
            interact_status (pd.DataFrame) : 
                negative interaction data with column user_name, list of negative item,
                and 100 sample of negative item
        """
        logger.info("STARTING SAMPLING NEGATIVE INTERACTION DATA")
        interact_status = self.mapped_data.groupby(self.user_name)[self.item_name].apply(set).reset_index().rename(columns={self.item_name: 'interacted_items'})
        # use netagive_items in training data
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        # use negatives_samples in evaluation data
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x),self.negative_interaction_sample))
        logger.info("FINISH SAMPLING NEGATIVE INTERACTION DATA")
        return interact_status[[self.user_name, 'negative_items', 'negative_samples']]

    def _split_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        """create train and test data from the latest interaction
        for every user
        Args:
        Returns:
            train_data (pd.DataFrame) :
                train data frame
            test_data (pd.DataFrame) : 
                test data frame
        """
        logger.info("START SPLITTING TRAIN AND TEST DATA")
        rank_timestamp = deepcopy(self.mapped_data)
        rank_timestamp['rank_latest'] = rank_timestamp.groupby([self.user_name])[self.timestamp_name].rank(method='first', ascending=False)
        test_data = rank_timestamp[rank_timestamp['rank_latest'] == 1]
        train_data = rank_timestamp[rank_timestamp['rank_latest'] > 1]

        # all user in test data must be exists in train data
        # means user with only one interaction is forbidden
        assert test_data[self.user_name].nunique() == train_data[self.user_name].nunique()

        logger.info("FINISH SPLITTING TRAIN AND TEST DATA")
        return train_data.drop(columns='rank_latest'), test_data.drop(columns='rank_latest')
    
    def create_train_data(self,num_negatives : int, batch_size : int) -> object:
        """Method to add positive train data with sampled negative data
        Args:
            num_negatives (int):
                number of negative sample for each positive interactions
            batch_size (int):
                batch size for the data loader
        Returns:
            object: 
                DataLoader for training
        """
        users, items, interaction = [], [], []
        train_data = pd.merge(self.train_data, self.negatives_interaction[[self.user_name, 'negative_items']], on=self.user_name)
        train_data['negatives'] = train_data['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
        for _, row in train_data.iterrows():
            users.append(int(row[self.user_name]))
            items.append(int(row[self.item_name]))
            interaction.append(float(row[self.interaction_name]))
            for i in range(num_negatives):
                users.append(int(row[self.user_name]))
                items.append(int(row['negatives'][i]))
                interaction.append(float(0)) # negative samples get 0 rating
        dataset = UserItemInteractionDataset(user_tensor = torch.LongTensor(users),
                                             item_tensor = torch.LongTensor(items),
                                             interaction_tensor = torch.FloatTensor(interaction))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def create_evaluation_data(self) -> list:
        """Method to create the evaluation data

        Returns:
            [list of tensor]: 
                tensor[0] : tensor of test data user_id positive interaction
                tensor[1] : tensor of test data item_id positive interaction
                tensor[2] : tensor of test data user_id negative interaction
                tensor[3] : tensor of test data item_id negative interaction
        """
        logger.info("START CREATING EVALUATION DATA")
        test_data = pd.merge(self.test_data, self.negatives_interaction[[self.user_name, 'negative_samples']], on=self.user_name)
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for _, row in test_data.iterrows():
            test_users.append(int(row[self.user_name]))
            test_items.append(int(row[self.item_name]))
            for i in range(len(row['negative_samples'])):
                negative_users.append(int(row[self.user_name]))
                negative_items.append(int(row['negative_samples'][i]))
        logger.info("FINISH CREATING EVALUATION DATA")
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
    

class UserItemInteractionDataset(Dataset):
    """Train Dataset contain User, Item, Interaction
       from positive data  and negative data"""
    def __init__(self,
                 user_tensor : torch.LongTensor,
                 item_tensor : torch.LongTensor,
                 interaction_tensor : torch.FloatTensor):
        """Initialization dataset
        Args:
            user_tensor (torch.LongTensor):
                tensor from user data (positive and negative)
            item_tensor (torch.LongTensor):
                tensor from item data (positive and negative)
            interaction_tensor (torch.FloatTensor):
                tensor from interaction data (positive and negative)
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.interaction_tensor = interaction_tensor

    def __getitem__(self, index : int):
        """Dataset indexing
        Args:
            index (int):
                index for slicing the dataset
        Returns:
            tensor tuple :
                tensor_user, tensor_item, tensor_interation
        """
        return self.user_tensor[index], self.item_tensor[index], self.interaction_tensor[index]
    
    def __len__(self):
        """Len method for the dataset
        Returns:
            len dataset : dataset rows
        """
        return self.user_tensor.size(0)