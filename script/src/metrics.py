import math
import pandas as pd

class MetronAtk(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None
    
    @property
    def top_k(self):
        return self._top_k
    
    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects
    
    @subjects.setter
    def subjects(self, subjects : list) -> pd.DataFrame:
        """Method to create full_df and rank each item id for 
           every user listed

            Args:
            subjects (list):
                list of test data tensor, positive interaction and 
                negative interaction

            Returns:
            full_df (pd.DataFrame):
                DataFrame consist of positive and negative item
                for evaluation
        """
        assert isinstance(subjects, list)    # assert if the input is list

        # get the test data tensor
        positive_users, positive_items, positive_interactions = subjects[0], subjects[1], subjects[2]
        negative_users, negative_items, negative_interactions = subjects[3], subjects[4], subjects[5]

        # create data frame with positive data
        positive_df = pd.DataFrame({'user': positive_users,
                                    'positive_item': positive_items,
                                    'positive_interaction' : positive_interactions})
        
        # create data frame with negative data
        full_df = pd.DataFrame({'user': negative_users + positive_users,
                                'full_item': negative_items + positive_items,
                                'full_interaction' : negative_interactions + positive_interactions})
        
        # rank each item for each user based on the calculated interaction
        full_df = pd.merge(full_df, positive_df, on=['user'], how='left')
        full_df['rank'] = full_df.groupby('user')['full_interaction'].rank(method='first', ascending=False)
        full_df.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full_df    

    def calculate_hit_ratio(self) -> float:
        """Calculate hit ratio 

        Returns:
            Hit Ratio (float):
                Ratio from predicted
        """
        # get top_k item for each user after ranked
        full_df, top_k = self._subjects, self._top_k
        top_k_df = full_df[full_df['rank']<=top_k]

        # check if the top_k data is in positive interaction
        test_in_top_k = top_k_df[top_k_df['positive_item'] == top_k_df['full_item']]  # golden items hit in the top_K items

        # calculate the ratio of user which positive item
        # is in top_k and all user
        return len(test_in_top_k) * 1.0 / full_df['user'].nunique()

    def calculate_ndcg(self) -> float:
        """Calculate NDCG

        Returns:
            _NDCG (float): _description_
        """
        # get top_k item for each user after ranked
        full_df, top_k = self._subjects, self._top_k
        top_k_df = full_df[full_df['rank'] <= top_k]

        # check if the top_k data is in positive interaction
        test_in_top_k = top_k_df[top_k_df['positive_item'] == top_k_df['full_item']]

        # calculate NDCG
        ndcg_list = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        return ndcg_list.sum() * 1.0 / full_df['user'].nunique()