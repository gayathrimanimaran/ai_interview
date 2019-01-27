import argparse
import sys
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils import success_, get_deadline_year, preprocess
from Models import LGBMModel
from sklearn.metrics import accuracy_score
import numpy

class Application:
    """
    Application
    ==========
    This class is runner class for the entire application.  It accepts certain command line parameters, then invokes the
    methods of the class you will create.

    Example run command:
    python application.py -f file_path_to_data
    """

    def __init__(self, file_path):
        """
        :param file_path (str): the path of the csv to read in
        """
        
        self.file_path = file_path
        if self.file_path is None:
            raise Exception("Must specify file path to the data")

    def _read_data(self, file_path):
        """
        Perform any removal of columns of data here

        :param file_path (str): the path of the csv to read in
        :return (obj:`(pd.DataFrame, list<float>, pd.DataFrame, list<float>`): a dataframe for the train data, train labels, 
        a dataframe for the test data, test labels
        """

        train_size = .75
        data_frame = pd.read_csv(file_path)
        
        #Feature Engineering an extra column
        data_frame['success_probability'] = success_(data_frame['usd_goal_real'].values, data_frame['usd_pledged_real'].values)
        data_frame['deadline_year'] = data_frame['deadline'].apply(lambda x: get_deadline_year(x))
        
        # Splitting the data into train set and test set
        split_size = int(data_frame.shape[0] * train_size)
        train_data_frame = data_frame[:split_size]
        test_data_frame = data_frame[split_size:]

        # Getting the labels
        train_labels = train_data_frame['state']
        test_labels = test_data_frame['state']

        return train_data_frame, train_labels, test_data_frame, test_labels

    def run(self):
        train_data_frame, train_labels, test_data_frame, test_labels = self._read_data(self.file_path)

        ######
        #label_encoder for encoding target labels
        label_encoder = LabelEncoder()
        label_encoder.fit(train_data_frame['state'].values)
        
        #Preprocess the data
        train_data, train_labels, test_data, test_labels = preprocess(train_data_frame,
                                                                      train_labels,
                                                                      test_data_frame,
                                                                      test_labels,
                                                                      label_encoder)
        
        # Here you put the model you will be using from the class you created
        model = LGBMModel()
        
        #training the model
        model.train(train_data, train_labels)
        
        #predicting the values for the test data
        predictions = model.predict(test_data)
        
        #Getting the original names of the labels
        prediction_names = label_encoder.inverse_transform(predictions)
        
        #Saving the error value in a text file
        model._save_predictions(test_data_frame['ID'].values, test_data_frame['name'].values, prediction_names)
        
        #Printing the error value
        print('\nAccuracy = {}'.format(accuracy_score(test_labels, predictions)))


"""
Entrance point for execution
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model trainer for starter project')
    parser.add_argument("-f", "--file_path", help = "where to read in the data", default = None)
    args = parser.parse_args(sys.argv[1:])

    app = Application(
        file_path = args.file_path
    )
    app.run()
