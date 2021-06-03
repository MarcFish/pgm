import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import argparse
import functools
import numpy as np

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

data = np.concatenate([np.random.normal(loc=0.0, scale=10.0, size=100), np.random.normal(loc=10.0, scale=5.0, size=200)])

Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
def model():
    cp = yield Root(tfd.Dirichlet(concentration=np.ones(2) * 0.5))
    c = tfd.Categorical(probs=cp)
    mu = yield Root(tfd.Normal(loc=tf.zeros(2, dtype=tf.float64), scale=tf.ones(2, dtype=tf.float64) * 10.))
    std = yield Root(tfd.Uniform(low=tf.zeros(2, dtype=tf.float64), high=tf.ones(2, dtype=tf.float64) * 100.))
    obs = yield tfd.Sample(tfd.MixtureSameFamily(
        mixture_distribution=c,
        components_distribution=tfd.Normal(
            loc=mu,  # One for each component.
            scale=std)), sample_shape=data.shape[0])
model = tfd.JointDistributionCoroutine(model)

def joint_log_prob(data, cp, mu, std):
  r_cp = tfd.Dirichlet(concentration=np.ones(2) * 0.5)
  c = tfd.Categorical(probs=cp)
  r_mu = tfd.Normal(loc=tf.zeros(2, dtype=tf.float64), scale=tf.ones(2, dtype=tf.float64) * 10.)
  r_std = tfd.Uniform(low=tf.zeros(2, dtype=tf.float64), high=tf.ones(2, dtype=tf.float64) * 100.)
  r_obs = tfd.MixtureSameFamily(
        mixture_distribution=c,
        components_distribution=tfd.Normal(
            loc=mu,
            scale=std))
  return r_cp.log_prob(cp) + r_mu.log_prob(mu) + r_std.log_prob(std) + tf.reduce_sum(r_obs.log_prob(data))
# def joint_log_prob(data, cp, mu, std):
#     model.log_prob(cp, mu, std, data)
unnormalized_posterior_log_prob = functools.partial(joint_log_prob, data)

initial_state = model.sample()
unconstraining_bijectors = [
    tfb.Softplus(),
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