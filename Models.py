import warnings
from lightgbm import LGBMClassifier
from base_model import BaseModel
from sklearn.utils.validation import DataConversionWarning
import numpy as np
import pandas as pd


class LGBMModel(BaseModel):
    """
    This is an example model that extends the base model.
    """

    def __init__(self):
        warnings.simplefilter('ignore', DataConversionWarning)

        super().__init__(model = LGBMClassifier())

    def train(self, training_dataset, training_labels):
        """
        This is the base "train" method.  It expects train/validation datasets and labels and
        uses them to train the underlying learning algorithm.

        :param training_dataset (obj:`pd.DataFrame`): A dataframe consisting of training data records
        :param training_labels (obj:`list<float>`): A list of labels ordered respective to the data in the training dataset
        """

        processed_data = self._preprocess_data(training_dataset)
        self.model.fit(processed_data, training_labels)

    def predict(self, dataframe):
        """
        This takes data and feeds it to the model to produce predictions

        :param dataset (obj:`pd.DataFrame`): Data used to create predictions
        :return (obj:'list<float>`): A list of all the predictions from the model
        """

        processed_data = self._preprocess_data(dataframe)
        predicted_values = self.model.predict(processed_data)
        return predicted_values

    def _preprocess_data(self, dataset):
        """
        Performs preprocessing operations on the data

        :param dataset (obj:`pd.DataFrame`): the data to be transformed
        :return (obj:`pd.DataFrame`): processed data
        """

        # here we are simply going to impute Nans with 0
        return dataset
    
    def _save_predictions(self, ID, name, predictions):
        '''
        prints the predictions in a csv file
        
        param:
            predictions: the labels of the test dataset
        '''
        project_id = np.reshape(ID, (-1,1))
        data = np.append(project_id, np.reshape(name, (-1,1)), axis = 1)
        data = np.append(data, np.reshape(predictions, (-1,1)), axis = 1)
        
        test = pd.DataFrame(data, columns = ['ID','name', 'predicted_state'])
        test.to_csv('test_predictions.csv', index = False, mode = 'w+')