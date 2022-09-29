#!/usr/bin/env python3.5


from __future__ import absolute_import, print_function
"""Defines NNMF models."""
# Third party modules
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer,l1_regularizer
from collections import defaultdict
from .utils import  build_mlp, get_kl_weight
import numpy as np
class _VABMFBase(object):
    def __init__(self, num_users, num_items, Fs_D=40, dims=[40,20,10],
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}, model_filename='model/nnmf.ckpt'):
        self.num_users = num_users
        self.num_items = num_items
        self.Fs_D = Fs_D
        self.hidden_units_per_layer = dims
        self.latent_normal_init_params = latent_normal_init_params
        self.model_filename = model_filename
        self.k=10
        # Internal counter to keep track of current epoch
        self._epochs = 0

        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None,10])
        
        
        # Call methods to initialize variables and operations (to be implemented by children)
        self._init_vars()
        self._init_ops()
        
        
        # RMSE
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.r, self.r_target))))

    def _init_vars(self):
        raise NotImplementedError

    def _init_ops(self):
        raise NotImplementedError

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _train_iteration(self, data, additional_feed=None):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings =data['rating']
        y = np.expand_dims(ratings, axis=1)
        s=np.ones((1,10))
        ratings=np.matmul(y,s)

      
        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target:ratings}

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._epochs += 1

    def train_iteration(self, data):
        self._train_iteration(data)

    def eval_loss(self, data):
        raise NotImplementedError

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']
        y = np.expand_dims(ratings, axis=1)
        s=np.ones((1,10))
        ratings=np.matmul(y,s)

     
        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        return self.sess.run(self.rmse, feed_dict=feed_dict)

    def predict(self,data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        rating =self.sess.run(self.r, feed_dict={self.user_index: user_ids, self.item_index: item_ids})
        return rating
    
    def precision_recall_at_k(self,predictions, threshold=3.5):
      '''Return precision and recall at k metrics for each user.'''

      # First map the predictions to each user.
      user_est_true = defaultdict(list)
      for uid,_, true_r,est in predictions:
          user_est_true[uid].append((est, true_r))

      precisions = dict()
      recalls = dict()
      for uid, user_ratings in user_est_true.items():
 
          # Sort user ratings by estimated value
          user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
          n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

          # Number of recommended items in top k
          n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:self.k])

          # Number of relevant and recommended items in top k
          n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:self.k])

          # Precision@K: Proportion of recommended items that are relevant
          precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

          # Recall@K: Proportion of relevant items that are recommended
          recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

      return precisions, recalls





class VABMf(_VABMFBase):
    
    def __init__(self, *args, **kwargs):
        self.num_latent_samples = 1
        self.num_data_samples = 3
        self.lambda_u = 0.1
        self.lambda_v = 10
        self.lambda_r = 1
        self.a = 1
        self.b = 0.001
        self.M = 300
        self.n_epochs = 100
        self.max_iter = 1
        self.lr = 0.001  #0.001 for 100k
        self.belta = 0.5
        if 'lam' in kwargs:
            self.lam = float(kwargs['lam'])
            del kwargs['lam']
        else:
            self.lam = 0.01   #0.01 for 100k

        if 'r_var' in kwargs:
            self.r_var = float(kwargs['r_var'])
            del kwargs['r_sigma']
        else:
            self.r_var = 1.0

       

        if 'kl_full_epoch' in kwargs:
            self.kl_full_epoch = int(kwargs['kl_full_epoch'])
            del kwargs['kl_full_epoch']
        else:
            self.kl_full_epoch = 100 # something like: max_epochs/3

        if 'anneal_kl' in kwargs:
            self.anneal_kl = bool(kwargs['anneal_kl'])
        else:
            self.anneal_kl = True

        super(VABMF, self).__init__(*args, **kwargs) 
   
    def _init_vars(self):
        
        
        # Latents
       
        self.U_mu=tf.Variable(tf.truncated_normal([self.num_users, self.Fs_D], **self.latent_normal_init_params))
        self.U_log_var= tf.Variable(tf.random_uniform([self.num_users, self.Fs_D], minval=0.0, maxval=0.5))

        self.V_mu = tf.Variable(tf.truncated_normal([self.num_items, self.Fs_D], **self.latent_normal_init_params))
        self.V_log_var = tf.Variable(tf.random_uniform([self.num_items, self.Fs_D], minval=0.0, maxval=0.5))

        
        # Lookups
        self.U_mu_lu = tf.nn.embedding_lookup(self.U_mu, self.user_index)
        U_log_var_lu = tf.nn.embedding_lookup(self.U_log_var, self.user_index)

       
        self.V_mu_lu = tf.nn.embedding_lookup(self.V_mu, self.item_index)
        V_log_var_lu = tf.nn.embedding_lookup(self.V_log_var, self.item_index)


        # Posterior (q) - note this handles reparameterization for us
        q_U =tf.contrib.distributions.MultivariateNormalDiag(self.U_mu_lu,tf.sqrt(tf.exp(U_log_var_lu)))
        
        q_V = tf.contrib.distributions.MultivariateNormalDiag(self.V_mu_lu,
               tf.sqrt(tf.exp(V_log_var_lu)))

        # Sample
        self.U = q_U.sample()
        self.V = q_V.sample()


        # MLP ("f")
              
        
        self.f_input_layer = tf.concat(axis=1, values=[self.U_mu_lu, self.V_mu_lu])
        

        x_recon, self.mlp_weights,self.z_mean,self.z_log_sigma_sq,self.reg_loss,self.z = build_mlp(self.hidden_units_per_layer,self.Fs_D,self.f_input_layer)
        self.r = self.z


        # For KL annealing
        self.kl_weight = tf.placeholder(tf.float32) if self.anneal_kl else tf.constant(1.0, dtype=tf.float32)

    def _init_ops(self):
        KL_all = tf.reduce_mean(tf.reduce_sum(0.5 * (-self.z_log_sigma_sq + tf.exp(self.z_log_sigma_sq) + self.z_mean**2 - 1), axis=1))

        log_prob = -(1/(2.0*self.r_var))*tf.reduce_sum(tf.square(tf.subtract(self.r_target, self.r)), reduction_indices=[0])
        reg = l2_regularizer(self.lam)
        self.reg_var = apply_regularization(reg,self.mlp_weights )
        neg_elbo = log_prob-(self.reg_var*KL_all)
        #self.loss = -neg_elbo
        #tf.add_n
        reconstruction_loss = tf.reduce_sum(tf.square(tf.subtract(self.r_target, self.r)), reduction_indices=[0])
      
        reg1=tf.nn.l2_loss(self.U)+tf.nn.l2_loss(self.V)
        #reg = l1_regularizer(self.lam)
        
        self.loss =-neg_elbo+ reconstruction_loss +(reg1*self.lam)+(self.reg_var*self.reg_loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        train_step=self.optimizer.minimize(self.loss)
       
        self.optimize_steps = [train_step ]

    def train_iteration(self, data):
        additional_feed = {self.kl_weight: get_kl_weight(self._epochs, on_epoch=self.kl_full_epoch)} if self.anneal_kl \
            else {}
        super(VABMF, self)._train_iteration(data, additional_feed=additional_feed)

    def eval_loss(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']
        y = np.expand_dims(ratings, axis=1)
        s=np.ones((1,10))
        ratings=np.matmul(y,s)

      
        feed_dict = {self.user_index: user_ids, self.item_index: item_ids,self.r_target:ratings,
                     self.kl_weight: get_kl_weight(self._epochs, on_epoch=self.kl_full_epoch)}
        return self.sess.run(self.loss, feed_dict=feed_dict)



