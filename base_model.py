import warnings

from sklearn.utils.validation import DataConversionWarning


class BaseModel(object):
    """
    This is the base model class that your algorithm will implement.
    An example of this would be having LinearRegression(BaseModel): class that implements these methods.
    """

    def __init__(self, model = None):
        """
        :param model (object): a model to train on 
        """
        
        warnings.simplefilter('ignore', DataConversionWarning)

        self.model = model   # Sub-classes must pass the model param when extending BaseModel

    def train(self, training_dataset, training_labels):
        """
        This is the base "train" method.  It expects train/validation datasets and labels and
        uses them to train the underlying learning algorithm.

        :param training_dataset (obj:`pd.DataFrame`): A dataframe consisting of training data records
        :param training_labels (obj:`list<float>`): A list of labels ordered respective to the data in the training dataset
        """

        # Make sure to preprocess the data
        raise NotImplementedError()

    def predict(self, dataframe):
        """
        This takes data and feeds it to the model to produce predictions

        :param dataset (obj:`pd.DataFrame`): Data used to create predictions
        :return (obj:`list<float>`): A list of all the predictions from the model
        """

        # Make sure to preprocess the data
        raise NotImplementedError()

    def _preprocess_data(self, dataset):
        """
        Performs preprocessing operations on the data

        :param dataset (obj:`pd.DataFrame`): the data to be transformed
        :return (obj:`pd.DataFrame`): processed data
        """

        # you want to do some preprocessing here that will work with both train and test. A good idea here is to
        # check if your preprocessing method is fit.
        raise NotImplementedError()