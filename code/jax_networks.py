import haiku as hk
import copy
import jax
import jax.numpy as jnp
import numpy as onp
import json
import optax
from functools import partial
import os
import sys

sys.path.append((os.path.dirname(os.path.dirname(__file__))))

from utils import *


def VANO(config):
    encoder_config = config["encoder_config"]
    decoder_config = config["decoder_config"]
    latent_dim = config["latent_dim"]
    key = config["key"]
    
    # decoder has three subnetworks
    decoder_x_config = decoder_config["decoder_x_config"]
    decoder_z_config = decoder_config["decoder_z_config"]
    decoder_joint_network_config = decoder_config["decoder_joint_network_config"]

    def forward(input_):
        # only input_[0], the branch input has the function information

        # encoder forward
        encoder_output = MLP(encoder_config)(input_[0])
        mean = encoder_output[:, :latent_dim] # first half of output is mean
        logvar = encoder_output[:, latent_dim:] # second half of output is logvar
        # reparameterization trick
        rand_number = jax.random.uniform(key)
        z = mean + rand_number * logvar # latent representation
        
        # decoder output
        decoder_x_output = MLP(decoder_x_config)(input_[0])
        decoder_z_output = MLP(decoder_z_config)(z)

        # concat the locations representation and the latent dim representation
        decoder_joint_input = jnp.hstack([decoder_x_output, decoder_z_output])

        decoder_joint_output = MLP(decoder_joint_network_config)(decoder_joint_input)

        return decoder_joint_output, mean, logvar, z

    return forward

def DeepONetCartesianProd(config):
    branch_config = config["branch_config"]
    trunk_config = config["trunk_config"]

    def forward(input_):
        branch_forward = MLP(branch_config)
        trunk_forward = MLP(trunk_config)
        bias_param = hk.get_parameter("bias", shape=(1,), init=jnp.zeros)

        branch_pred = branch_forward(input_[0])
        trunk_pred = trunk_forward(input_[1])
        pred = jnp.matmul(branch_pred, trunk_pred.T) + bias_param
        return pred

    return forward

def DeepONetCartesianConvProd(config):
    branch_config = config["branch_config"]
    trunk_config = config["trunk_config"]

    def forward(input_):
        branch_forward = CNN(branch_config)
        trunk_forward = MLP(trunk_config)
        bias_param = hk.get_parameter("bias", shape=(1,), init=jnp.zeros)

        branch_pred = branch_forward(input_[0])
        trunk_pred = trunk_forward(input_[1])
        pred = jnp.matmul(branch_pred, trunk_pred.T) + bias_param
        return pred

    return forward

def CNN(config):
    if config["activation"] == "tanh":
        activation = jax.nn.tanh
    elif config["activation"] == "relu":
        activation = jax.nn.relu
    elif config["activation"] == "elu":
        activation = jax.nn.elu
    elif config["activation"] == "leaky_relu":
        activation = jax.nn.leaky_relu

    def forward(x):
        layers_ = []
        for i in range(len(config["layers"])):
            if "conv" in config["layers"][i]:
                _, num_channels, kernel_size, stride, padding = split_conv_string(config["layers"][i])
                layers_.append(hk.Conv2D(output_channels=num_channels, kernel_shape=kernel_size, stride=stride, padding=padding, name=config["name"]+f"_conv_{i}"))
            elif config["layers"][i] == "activation":
                layers_.append(activation)
            elif config["layers"][i] == "flatten":
                layers_.append(hk.Flatten())
            elif "linear" in config["layers"][i]:
                _, num_neurons = split_linear_string(config["layers"][i])
                layers_.append(hk.Linear(num_neurons, name=config["name"]+f"_linear_{i}"))
            else:
                raise f"Layer {config['layers'][i]} not configured"

        cnn = hk.Sequential(layers_)
        return cnn(x)

    return forward

def MLP(config):
    if config["activation"] == "tanh":
        activation = jax.nn.tanh
    elif config["activation"] == "relu":
        activation = jax.nn.relu
    elif config["activation"] == "elu":
        activation = jax.nn.elu
    elif config["activation"] == "leaky_relu":
        activation = jax.nn.leaky_relu

    if config.get("layer_sizes", None) is None:
        hidden_layers = [config["nodes"] for _ in range(config["num_hidden_layers"])]
        if config["nodes"] == 0 or config["num_hidden_layers"] == 0:
            layer_sizes = [config["output_dim"]]
        else:
            layer_sizes = hidden_layers + [config["output_dim"]]
    else:
        hidden_layers = config["layer_sizes"]
        layer_sizes = hidden_layers + [config["output_dim"]]

    def forward(x):
        mlp_module = hk.nets.MLP(output_sizes=layer_sizes, with_bias=config.get("use_bias", True), activation=activation, activate_final=config.get("last_layer_activate", False), name=config["name"])
        return mlp_module(x)

    return forward

def Linear(output_dim, use_bias=True):
    def forward(x):
        linear_module = hk.Linear(output_dim, with_bias=use_bias)
        return linear_module(x)

    return forward

def get_model(model_name, config):
    _MODELS = dict(
        mlp = MLP,
        linear = Linear,
        cnn = CNN,
        deeponet_cartesian_prod = DeepONetCartesianProd,
        deeponet_cartesian_conv_prod = DeepONetCartesianConvProd,
        vano = VANO
    )

    _USE_STATE = dict(
        mlp = False,
        linear = False,
        cnn = False,
        deeponet_cartesian_prod = False,
        deeponet_cartesian_conv_prod = False,
        vano = False
    )

    if model_name not in _MODELS.keys():
        raise NameError('Available keys:', _MODELS.keys())

    net_fn = _MODELS[model_name](config)

    if _USE_STATE[model_name]:
        net = hk.transform_with_state(net_fn)
    else:
        net = hk.transform(net_fn)

    return net.init, net.apply

