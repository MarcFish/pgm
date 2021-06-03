import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import argparse
import pandas as pd
import functools
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import mean_squared_error as mse

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float64

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='../data/train_rating.csv')
parser.add_argument("--test",type=str,default='../data/test_rating.csv')
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--user_size",type=int,default=6040)
parser.add_argument("--item_size",type=int,default=3706)
parser.add_argument("--lr_u",type=float,default=0.01)
parser.add_argument("--lr_i",type=float,default=0.01)
parser.add_argument("--embed_size",type=int,default=16)
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()

user_size = arg.user_size
item_size = arg.item_size

train = pd.read_csv(arg.file)
test = pd.read_csv(arg.test)
rating=coo_matrix((train.rating.values,(train.user.values,train.movie.values)),shape=(user_size,item_size)).todense().astype(np.float64)
test_rating=coo_matrix((test.rating.values,(test.user.values,test.movie.values)),shape=(user_size,item_size)).todense()
test_rating_coo = coo_matrix(test_rating)

Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
def model():
    lambda_u = yield Root(tfd.WishartTriL(df=tf.cast(arg.embed_size, tf.float64), scale_tril=tf.eye(tf.cast(arg.embed_size, tf.float64), dtype=tf.float64), input_output_cholesky=True))
    lambda_v = yield Root(tfd.WishartTriL(df=tf.cast(arg.embed_size, tf.float64), scale_tril=tf.eye(tf.cast(arg.embed_size, tf.float64), dtype=tf.float64), input_output_cholesky=True))
    mu_v = yield tfd.MultivariateNormalTriL(loc=0, scale_tril=2*tf.cast(lambda_v, tf.float64))
    mu_u = yield tfd.MultivariateNormalTriL(loc=0, scale_tril=2*tf.cast(lambda_u, tf.float64))
    u = yield tfd.Sample(tfd.MultivariateNormalTriL(loc=mu_u, scale_tril=tf.cast(lambda_u, tf.float64)), sample_shape=(user_size,))
    v = yield tfd.Sample(tfd.MultivariateNormalTriL(loc=mu_v, scale_tril=tf.cast(lambda_v, tf.float64)), sample_shape=(item_size,))
    r = tf.matmul(u, v, transpose_b=True)
    likelihood = yield tfd.Independent(tfd.Normal(loc=r, scale=2.0),reinterpreted_batch_ndims=2)

model = tfd.JointDistributionCoroutine(model)

def joint_log_prob(r,lu, lv, mv, mu, u, e):
  return model.log_prob(lu, lv, mv, mu, u,e,r)
unnormalized_posterior_log_prob = functools.partial(joint_log_prob, rating)

initial_state = model.sample()
unconstraining_bijectors = [
    tfb.CorrelationCholesky(),
    tfb.CorrelationCholesky(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
]
@tf.function
def sample():
  return tfp.mcmc.sample_chain(
    num_results=2,
    num_burnin_steps=500,
    current_state=initial_state[:-1],
    kernel=tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                 step_size=0.065,
                 num_leapfrog_steps=5),
            bijector=unconstraining_bijectors),
         num_adaptation_steps=1000))
r_ = sample()
U=tf.squeeze(tf.reduce_mean(r_[0][4],0))
I=tf.squeeze(tf.reduce_mean(r_[0][5],0))
r=tf.matmul(U,I, transpose_b=True).numpy()
rmse = np.sqrt(mse(r[test_rating_coo.nonzero()], test_rating_coo.data))
