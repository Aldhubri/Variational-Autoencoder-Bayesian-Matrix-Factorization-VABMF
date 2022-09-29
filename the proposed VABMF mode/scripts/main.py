#!/usr/bin/env python3.5
from __future__ import absolute_import, print_function
"""Trains/evaluates VAEMF models."""
import argparse, json, os
import tensorflow as tf
import pandas as pd
import sys
sys.path.append("..")
from vabmf.models import  VABMF
from vabmf.utils import chunk_df
import matplotlib.pyplot as plt
import random


def load_data(train_filename, valid_filename, test_filename, delimiter='\t', col_names=['user_id', 'item_id', 'rating']):
    """Helper function to load in/preprocess dataframes"""
    train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
    train_data['user_id'] = train_data['user_id'] - 1
    train_data['item_id'] = train_data['item_id'] - 1
    valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
    valid_data['user_id'] = valid_data['user_id'] - 1
    valid_data['item_id'] = valid_data['item_id'] - 1
    test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
    test_data['user_id'] = test_data['user_id'] - 1
    test_data['item_id'] = test_data['item_id'] - 1

    return train_data, valid_data, test_data

def train(model, sess, saver, train_data,test_data, valid_data, batch_size, max_epochs, use_early_stop, early_stop_max_epoch):
    # Print initial values
    train_rmse=[]
    valid_rmse=[]
    recall=[]
    batch = train_data.sample(batch_size) if batch_size else train_data
    train_error = model.eval_loss(batch)
    train_rmse.append(model.eval_rmse(batch))
    valid_rmse.append(model.eval_rmse(valid_data))
    print(" Train RMSE:",train_rmse[-1]," Valid RMSE:", valid_rmse[-1])
    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_epochs = 0
    for epoch in range(max_epochs):
        # Run (S)GD
        shuffled_df = train_data.sample(frac=1)
        batches = chunk_df(shuffled_df, batch_size) if batch_size else [train_data]

        for batch_iter, batch in enumerate(batches):
            model.train_iteration(batch)

            # Evaluate
            train_error = model.eval_loss(batch)
            train_rmse.append(model.eval_rmse(batch))
            valid_rmse.append(model.eval_rmse(valid_data))
            
            #est_rating=model.predict(batch)
            #est_rating=est_rating.reshape(75000 )
            #est_rating=[random.choice(est_rating) for _ in range(2000)]
            #data=pd.concat([valid_data, pd.DataFrame(est_rating)], axis=1)
            #data.columns = ['user_id', 'item_id','actual_r','pred_r']
            #test=data.to_records(index=False)
            #precisions, recalls = model.precision_recall_at_k(test, threshold=3.5)
            #recall.append(round(sum(rec for rec in recalls.values()) / len(recalls),7))
            
            #print('pers@: ',round((sum(prec for prec in precisions.values()) / len(precisions)),7),'recall@ : ',recall[-1],"Train RMSE:",train_rmse[-1]," Valid RMSE:", valid_rmse[-1])
            print(" Train RMSE:",train_rmse[-1]," Valid RMSE:", valid_rmse[-1])



        # Checkpointing/early stopping
        if use_early_stop:
            early_stop_epochs += 1
            if valid_rmse[-1] < prev_valid_rmse:
                prev_valid_rmse = valid_rmse[-1]
                early_stop_epochs = 0
                saver.save(sess, model.model_filename)
            elif early_stop_epochs == early_stop_max_epoch:
                print("Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse[-1]))
                return train_rmse,valid_rmse,recall
        else:
            saver.save(sess, model.model_filename)
        

def test(model, sess, saver, test_data, train_data=None, log=True):
    if train_data is not None:
        train_rmse = model.eval_rmse(train_data)
        if log:
            print("Final train RMSE: {}".format(train_rmse))

    test_rmse = model.eval_rmse(test_data)
    if log:
        print("Final test RMSE: {}".format(test_rmse))

    return test_rmse

if __name__ == '__main__':
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates VABMF models.')
    parser.add_argument('--model', metavar='MODEL_NAME',default='VABMF' ,type=str, help='the name of the model to use')
    parser.add_argument('--mode', metavar='MODE',default='train', type=str, help='the mode to run the program in')
    parser.add_argument('--train', metavar='TRAIN_INPUT_FILE', type=str, default='../data/ml-100k/split/u.data.train',
                        help='the location of the training set\'s input file')
    parser.add_argument('--valid', metavar='VALID_INPUT_FILE', type=str, default='../data/ml-100k/split/u.data.valid',
                        help='the location of the validation set\'s input file')
    parser.add_argument('--test', metavar='TEST_INPUT_FILE', type=str, default='../data/ml-100k/split/u.data.test',
                        help='the location of the test set\'s input file')
    parser.add_argument('--users', metavar='NUM_USERS', type=int, default=943, # ML 100K has 943 users
                        help='the number of users in the data set')
    parser.add_argument('--movies', metavar='NUM_MOVIES', type=int, default=1682, # ML 100K has 1682 movies
                        help='the number of movies in the data set')
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, default='{}',
                        help='JSON string containing model params')
    parser.add_argument('--delim', metavar='DELIMITER', type=str, default='\t',
                        help='the delimiter to use when parsing input files')
    parser.add_argument('--cols', metavar='COL_NAMES', type=str, default=['user_id', 'item_id', 'rating'],
                        help='the column names of the input data', nargs='+')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=5000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', default=False, action='store_true',
                        help='disable early stopping')
    parser.add_argument('--early-stop-max-epoch', metavar='EARLY_STOP_MAX_EPOCH', type=int, default=40,
                        help='the maximum number of epochs to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=500,
                        help='the maximum number of epochs to allow the model to train for')
    parser.add_argument('--hyperparam-search-size', metavar='HYPERPARAM_SEARCH_SIZE', type=int, default=50,
                        help='when in "select" mode, the number of times to sample for random search')

    # Parse args
    
    
    args = parser.parse_args()
    # Global args
    model_name = args.model
    mode = args.mode
    train_filename = args.train
    valid_filename = args.valid
    test_filename = args.test
    num_users = args.users
    num_items = args.movies
    model_params = json.loads(args.model_params)
    delimiter = args.delim
    col_names = args.cols
    batch_size = args.batch
    use_early_stop = not(args.no_early)
    early_stop_max_epoch = args.early_stop_max_epoch
    max_epochs = args.max_epochs

    if mode in ('train', 'test'):
        with tf.Session() as sess:
            # Define computation graph & Initialize
            print('Building network & initializing variables')
           
            if model_name == 'VABMF':
                model = VABMF(num_users, num_items, **model_params)
            else:
                raise NotImplementedError("Model '{}' not implemented".format(model_name))

            model.init_sess(sess)
            saver = tf.train.Saver()

            # Train
            if mode in ('train', 'test'):
                # Process data
                print("Reading in data")
                train_data, valid_data, test_data = load_data(train_filename, valid_filename, test_filename,
                    delimiter=delimiter, col_names=col_names)

                if mode == 'train':
                    # Create model directory, if needed
                    if not os.path.exists(os.path.dirname(model.model_filename)):
                        os.makedirs(os.path.dirname(model.model_filename))

                    # Train
                    rmse_train,rmse_test,recall=train(model, sess, saver, train_data,test_data, valid_data, batch_size=batch_size, max_epochs=max_epochs,
                          use_early_stop=use_early_stop, early_stop_max_epoch=early_stop_max_epoch)
                
                print('Loading best checkpointed model')
                
                print("\n\n Train RMSE:",rmse_train[-1]," Valid RMSE:", rmse_test[-1])

                #print('n recall :',recall[-1])
                with plt.style.context('dark_background'):
                  plt.plot(range(len(recall)),recall,color='g',label='@recall')
                  plt.plot(rmse_train,label='Training Data',color='w')
                  plt.plot(rmse_test,color='r',  label='Test Data')
                  plt.title('The MovieLens Dataset Learning Curve')
                  plt.xlabel('Number of Epochs')
                  plt.ylabel('RMSE -- Recall')
                  plt.legend()
                  plt.grid()
                  plt.figure(figsize=(20,40))
                plt.show()






