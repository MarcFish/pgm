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

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='../data/train_rating.csv')
parser.add_argument("--test",type=str,default='../data/test_rating.csv')
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--user_size",type=int,default=6040)
parser.add_argument("--item_size",type=int,default=3706)
parser.add_argument("--lr_u",type=float,default=0.01)
parser.add_argument("--lr_i",type=float,default=0.01)
parser.add_argument("--embed_size",type=int,default=128)
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()

user_max=4000
item_max=3000
user_size = arg.user_size
item_size = arg.item_size

train = pd.read_csv(arg.file)
test = pd.read_csv(arg.test)
rating=coo_matrix((train.rating.values,(train.user.values,train.movie.values)),shape=(user_size,item_size)).todense()[:user_max,:item_max]
test_rating=coo_matrix((test.rating.values,(test.user.values,test.movie.values)),shape=(user_size,item_size)).todense()[:user_max,:item_max]
test_rating_coo = coo_matrix(test_rating)

Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
def model():
  u = yield Root(tfd.Sample(tfd.Normal(loc=tf.cast(0, dtype), scale=0.2),sample_shape=[dim,user_size]))
  i = yield Root(tfd.Sample(tfd.Normal(loc=tf.cast(0, dtype), scale=0.2),sample_shape=[dim,item_size]))
  r = tf.matmul(u,i,adjoint_a=True)
  likelihood = yield tfd.Independent(
        tfd.Normal(loc=r, scale=0.2),
        reinterpreted_batch_ndims=2
    )

model = tfd.JointDistributionCoroutine(model)

def joint_log_prob(r,u,e):
  return model.log_prob(u,e,r)
unnormalized_posterior_log_prob = functools.partial(joint_log_prob, rating)

u0,i0,_ = mdl_ols_coroutine.sample()
initial_state = [u0,i0]
unconstraining_bijectors = [
    tfb.Identity(),
    tfb.Identity(),
]
@tf.function
def sample():
  return tfp.mcmc.sample_chain(
    num_results=5,
    num_burnin_steps=500,
    current_state=initial_state,
    kernel=tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                 step_size=0.065,
                 num_leapfrog_steps=5),
            bijector=unconstraining_bijectors),
         num_adaptation_steps=1000))

r_ = sample()
U=tf.squeeze(tf.reduce_mean(r_[0][0],0))
I=tf.squeeze(tf.reduce_mean(r_[0][1],0))
r_=tf.matmul(U,I,adjoint_a=True).numpy()
rmse = np.sqrt(mse(r_[test_rating_coo.nonzero()], test_rating_coo.data))

print("RMSE:{:.4f}".format(rmse))
with open(arg.result, 'a',encoding='utf-8',newline='') as f:
    f.write("MF,{:.4f}\n".format(rmse))
