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

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--num_topic", type=int, default=10)

arg = parser.parse_args()
doc = np.random.randint(low=0, high=2, size=(20, 30))
M = doc.shape[0]
N = doc.shape[1]
K = arg.num_topic
c = doc.sum(1)
def joint_log_prob(doc, theta, phi, z):
    rv_theta = tfd.Sample(tfd.Dirichlet(concentration=np.ones(arg.num_topic) * arg.alpha), sample_shape=[M])
    rv_phi = tfd.Sample(tfd.Dirichlet(concentration=np.ones(N) * arg.beta), sample_shape=[K])
    rv_z = tfd.Sample(tfd.Categorical(probs=theta), sample_shape=(10,))
    phiz = tf.nn.embedding_lookup(phi, rv_z)
    rv_w = tfd.Multinomial(total_count=c, probs=phiz)
    return rv_theta.log_prob(theta) + rv_phi.log_prob(phi) + rv_z.log_prob(z)
# 文档的表示方式应该为文档数量x词数量，其中的数值为该词在文档中出现的次数，增加超参，即每个文档中拥有的词数量
# def joint_log_prob(doc, theta, phi, z):
#     rv_theta = tfd.Sample(tfd.Dirichlet(concentration=np.ones(arg.num_topic) * arg.alpha), sample_shape=[M])
#     rv_phi = tfd.Sample(tfd.Dirichlet(concentration=np.ones(N) * arg.beta), sample_shape=[K])
#     rv_z = tfd.Categorical(probs=theta)
#     rv_z_ = tf.nn.embedding_lookup(phi, z)
#     rv_w = tfd.Multinomial(total_count=c, probs=rv_z_)
#     return rv_theta.log_prob(theta)+rv_phi.log_prob(phi)+rv_z.log_prob(z)+rv_w.log_prob(doc)
theta = tfd.Dirichlet(concentration=np.ones((M, K)) * arg.alpha).sample()  # doc-topic
phi = tfd.Dirichlet(concentration=np.ones((K, N)) * arg.beta).sample()  # topic-word
topic = tfd.Sample(tfd.Categorical(probs=theta), sample_shape=(10,)).sample()  # every doc have 10 topic
phiz = tf.nn.embedding_lookup(phi, topic)  # get topic-word dist for every doc-topic, shape is num_doc, 10, word
w = tfd.Multinomial(total_count=20, probs=phiz)  # get word dist for every topic for every doc
# z = tfd.Sample(tfd.Categorical(probs=theta), sample_shape=(N,)).sample()  # for every word sample a topic by doc-topic
# phiz = tf.nn.embedding_lookup(phi, z)
# w = tfd.Multinomial(total_count=c.astype(np.float64), probs=z)
# Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
# def model():
#     theta = yield Root(tfd.Dirichlet(concentration=np.ones((M, K)) * arg.alpha))  # doc-topic
#     phi = yield Root(tfd.Dirichlet(concentration=np.ones((K, N)) * arg.beta))  # topic-word
#     z = yield tfd.Multinomial(total_count=c, probs=theta)
#     z = tf.nn.embedding_lookup(phi, z)
#     w = yield tfd.Independent(tfd.Multinomial(total_count=c, probs=z),reinterpreted_batch_ndims=1)

# model = tfd.JointDistributionCoroutine(model)
#
# def joint_log_prob(doc, theta, phi, z):
#   return model.log_prob(theta, phi, z, doc)
# unnormalized_posterior_log_prob = functools.partial(joint_log_prob, doc)
#
# initial_state = model.sample()
# unconstraining_bijectors = [
#     tfb.Softplus(),
#     tfb.Softplus(),
#     tfb.Identity(),
# ]
# @tf.function
# def sample():
#   return tfp.mcmc.sample_chain(
#     num_results=2,
#     num_burnin_steps=500,
#     current_state=initial_state[:-1],
#     kernel=tfp.mcmc.SimpleStepSizeAdaptation(
#         tfp.mcmc.TransformedTransitionKernel(
#             inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
#                 target_log_prob_fn=unnormalized_posterior_log_prob,
#                  step_size=0.065,
#                  num_leapfrog_steps=5),
#             bijector=unconstraining_bijectors),
#          num_adaptation_steps=1000))
# r_ = sample()