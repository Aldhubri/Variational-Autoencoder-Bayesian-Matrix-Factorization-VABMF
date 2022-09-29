#!/usr/bin/env python3.5


from __future__ import absolute_import, print_function
import math
import tensorflow as tf



def activate(linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear': return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')
        elif name == 'Leaky ReLU':
            return tf.nn.leaky_relu(linear, name='encoded')
        elif name == 'elu':
            return tf.nn.elu(linear, name='encoded')
        elif name == 'softplus':
            return tf.nn.softplus(linear, name='encoded')



def build_mlp(dims,n_z,f_input_layer):
    mlp_weights=[]
    num_f_inputs = f_input_layer.get_shape().as_list()[1]
   
        
    activations=['sigmoid', 'softmax','linear','tanh','relu','Leaky ReLU','elu','softplus']
    kongzhi = 0.5
    batch_size=1
    n_z=5
    """Builds a feed-forward NN (MLP) with 4 hidden layers."""
    
    

    with tf.variable_scope("inference_network"):
      rec = {
        'W1': tf.get_variable('W1',[num_f_inputs, dims[0]],initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
        'b1': tf.get_variable('b1', [dims[0]],initializer=tf.constant_initializer(0.0),dtype=tf.float32),
        'W2': tf.get_variable('W2',[dims[0], dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
        'b2': tf.get_variable("b2",[dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        'W3': tf.get_variable('W3',[dims[1],dims[2] ], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
        'b3': tf.get_variable("b3", [dims[2]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        'W_z_mean': tf.get_variable("W_z_mean", [dims[2], n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
        'b_z_mean': tf.get_variable("b_z_mean", [n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [dims[2],n_z], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
        'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        
         }
         
    with tf.variable_scope("generation_network"):
            gen = {'W3': tf.get_variable('W3',[n_z,dims[2]] ,initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b3': tf.get_variable('b3',dims[2],initializer=tf.constant_initializer(0.0),dtype=tf.float32),
                'W2': tf.get_variable('W2',[dims[2], dims[1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2",[dims[1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W1': tf.get_variable('W1',[dims[1], dims[0]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1': tf.get_variable("b1", [dims[0]], initializer=tf.constant_initializer(0.0), dtype=tf.float32),   
                'W_x': tf.get_variable("W_x", [dims[0],num_f_inputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_x': tf.get_variable("b_x", [num_f_inputs], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                
                }
        
     
    mlp_weights+=[rec['W1'], rec['b1'], rec['W2'], rec['b2'],rec['W3'], rec['b3'],
                  rec['W_z_mean'], rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
                
    reg_loss = tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2']) + tf.nn.l2_loss(rec['W3'])

    h1 = activate(tf.matmul(f_input_layer, rec['W1']) + rec['b1'], activations[5])
    h2 = activate(tf.matmul(h1,rec['W2']) + rec['b2'], activations[5])
    h3 = activate(tf.matmul(h2,rec['W3'] ) + rec['b3'], activations[5])
    
    z_mean = tf.matmul(h3, rec['W_z_mean']) + rec['b_z_mean']
    z_log_sigma_sq = tf.matmul(h3, rec['W_z_log_sigma']) + rec['b_z_log_sigma']
      
    z_mean = kongzhi * z_mean
    z_log_sigma_sq=z_log_sigma_sq*kongzhi

    sigma= z_log_sigma_sq[:,num_f_inputs:]
    eps = tf.random_normal(tf.shape(z_log_sigma_sq))
    z = z_mean + tf.exp(z_log_sigma_sq)* eps
    mlp_weights += [gen['W1'], gen['b1'], gen['W2'], gen['b2'],gen['W3'], gen['b3'],
                  gen['W_x'], gen['b_x']]
    reg_loss += tf.nn.l2_loss(gen['W1']) + tf.nn.l2_loss(gen['W2']) + tf.nn.l2_loss(gen['W3'])

    
    h3 = activate(tf.matmul(z, gen['W3'])+gen['b3'] , activations[5])
    h2 = activate(tf.matmul(h3,gen['W2'])+gen['b2'] , activations[5]) 
    x_recon = activate((tf.matmul(h2, gen['W1'])+gen['b1']) ,activations[5])

    return x_recon, mlp_weights,z_mean,z_log_sigma_sq,reg_loss, z  


def get_kl_weight(curr_iter, on_epoch=100):
    """Outputs sigmoid scheduled KL weight term (to be fully on at 'on_epoch')"""
    return 1.0 / (1 + math.exp(-(25.0 / on_epoch) * (curr_iter - (on_epoch / 2.0))))


def chunk_df(df, size):
    """Splits a Pandas dataframe into chunks of size `size`.

    See here: https://stackoverflow.com/a/25701576/1424734
    """
    return (df[pos:pos + size] for pos in range(0, len(df), size))
