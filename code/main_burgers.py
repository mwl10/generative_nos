import torch
from functools import partial
import matplotlib.pyplot as plt
import time
import copy
from collections import defaultdict
import json
from scipy import io
import optax
from jax.example_libraries import optimizers
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

sys.path.append((os.path.dirname(os.path.dirname(__file__))))

from jax_networks import get_model
from utils import *


def get_data():
    sub_x = 2 ** 6
    sub_y = 2 ** 6

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat(str(dataset_folder)+"burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(onp.float32)
    y_data = data["u"][:, ::sub_y].astype(onp.float32)
    x_branch_train = x_data[:N, :]
    y_train = y_data[:N, :]
    x_branch_test = x_data[-N_test:, :]
    y_test = y_data[-N_test:, :]

    s = 2 ** 13 // sub_y  # total grid size divided by the subsampling rate
    grid = onp.linspace(0, 1, num=2 ** 13)[::sub_y, None]

    x_branch_train = x_branch_train.astype(dtype)
    grid = grid.astype(dtype)
    x_branch_test = x_branch_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)
    return x_train, y_train, x_test, y_test


def train_func(train_info):
    # to store results
    logged_results = defaultdict(list)

    @jax.jit
    def loss(all_params, input_, y):
        # vano loss
        pred, mean, logvar, z = model_forward(all_params, None, input_)

        # reconstruction loss
        loss = jnp.square(pred - input_[0]).mean()

        # KL loss
        loss += kl_weight * jnp.mean(-0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), 1), 0)

        return loss

    @jax.jit
    def step(all_params, opt_state, input_, y):
        # grads is for all grads
        loss_val, grads = jax.value_and_grad(loss)(all_params, input_, y)
        updates, opt_state = opt.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)
        return all_params, opt_state, loss_val

    # hyperparameters and dataset
    print_interval = train_info["print_interval"]
    print_bool = train_info["print_bool"]
    epochs = train_info["epochs"]
    train_input = train_info["train_input"]
    test_input = train_info["test_input"]
    Y = train_input[0]
    Y_test = test_input[0]
    dummy_input = train_info["dummy_input"]

    model_key = jax.random.PRNGKey(train_info["seed"])

    model_init, model_forward = get_model(train_info["model_name"], train_info["model_config"])
    model_forward = jax.jit(model_forward)
    # initializing model parameters
    all_params = model_init(model_key, dummy_input)

    if train_info["schedule_choice"] == "warmup_cosine_decay":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value = train_info["lr_dict"]["init_lr"],
            peak_value = train_info["lr_dict"]["peak_lr"],
            warmup_steps = train_info["lr_dict"]["warmup_steps"],
            decay_steps = train_info["lr_dict"]["decay_steps"]
        )
    elif train_info["schedule_choice"] == "inverse_time_decay":
        schedule = optimizers.inverse_time_decay(
            step_size = train_info["lr_dict"]["peak_lr"],
            decay_steps = train_info["lr_dict"]["decay_steps"],
            decay_rate = train_info["lr_dict"]["decay_rate"],
            staircase = train_info["lr_dict"].get("staircase", False)
        )
    elif train_info["schedule_choice"] == "no_schedule":
        schedule = train_info["peak_lr"]
    else:
        raise "Unknown learning rate schedule"

    if train_info["opt_choice"] == "adam":
        opt = optax.adam(learning_rate=schedule)
    elif train_info["opt_choice"] == "adamw":
        opt = optax.adamw(learning_rate=schedule, weight_decay=train_info["lr_dict"]["weight_decay"])

    # initializing optimizer state with parameters
    opt_state = opt.init(all_params)

    # warmup
    _, _, _ = step(all_params, opt_state, train_input, Y)
    print("\nStarting training")

    for i in range(epochs):
        start = time.perf_counter()
        all_params, opt_state, loss_val = step(all_params, opt_state, train_input, Y)
        epoch_time = time.perf_counter() - start

        logged_results["training_loss"].append(loss_val.item())
        logged_results["training_epoch_time"].append(epoch_time)

        """
        we do inference only for the final time-step
        """

        # train inference warmup
        _ = model_forward(all_params, None, train_input)
        start = time.perf_counter()
        train_pred, _, _, _ = model_forward(all_params, None, train_input)
        train_inference_time = time.perf_counter() - start

        # test inference warmup
        _ = model_forward(all_params, None, test_input)
        start = time.perf_counter()
        test_pred, _, _, _ = model_forward(all_params, None, test_input)
        test_inference_time = time.perf_counter() - start

        assert train_pred.shape == Y.shape, f"train pred = {train_pred.shape}, Y = {Y.shape}"
        assert test_pred.shape == Y_test.shape, f"test pred = {test_pred.shape}, Y_test = {Y_test.shape}"

        test_l2_error = jnp.linalg.norm(test_pred - Y_test) / jnp.linalg.norm(Y_test)
        test_linf_error = jnp.linalg.norm(test_pred - Y_test, ord=jnp.inf) / jnp.linalg.norm(Y_test, ord=jnp.inf)
        test_mse_error = jnp.power(test_pred - Y_test, 2).mean()

        train_l2_error = jnp.linalg.norm(train_pred - Y) / jnp.linalg.norm(Y)
        train_linf_error = jnp.linalg.norm(train_pred - Y, ord=jnp.inf) / jnp.linalg.norm(Y, ord=jnp.inf)
        train_mse_error = jnp.power(train_pred - Y, 2).mean()

        logged_results["train_inference_time"].append(train_inference_time)
        logged_results["train_l2_error"].append(train_l2_error.item())
        logged_results["train_linf_error"].append(train_linf_error.item())
        logged_results["train_mse_error"].append(train_mse_error.item())
        logged_results["test_inference_time"].append(test_inference_time)
        logged_results["test_l2_error"].append(test_l2_error.item())
        logged_results["test_linf_error"].append(test_linf_error.item())
        logged_results["test_mse_error"].append(test_mse_error.item())

        if i % print_interval == 0 and print_bool:
            print("="*15)
            print(f"Epoch {i}:")
            print(f"Loss: {loss_val}")
            print(f"Time: {epoch_time}")
            print(f"\nTrain relative L2 error: {train_l2_error}")
            print(f"Test relative L2 error: {test_l2_error}")
            print(f"Train relative Linf error: {train_linf_error}")
            print(f"Test relative Linf error: {test_linf_error}")
            print(f"Train MSE: {train_mse_error}")
            print(f"Test MSE: {test_mse_error}")
            print(f"Train inference time: {train_inference_time}")
            print(f"Test inference time: {test_inference_time}\n")

    return logged_results, all_params

def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__=="__main__":
    dataset_folder = fstr("../data/{problem}/")

    save_results_bool = True
    print_bool = True

    problem = "Burgers"
    opt_choice = "adam"
    schedule_choice = "inverse_time_decay"

    print_interval = 100
    epochs = 50000
    dtype = "float32"
    # activations
    activation_choice = "leaky_relu"

    m = 128
    N = 1000
    N_test = 200

    lr_dict = dict(
        peak_lr = 1e-3,
        decay_steps = epochs // 5,
        decay_rate = 0.5
    )

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    seed = 0

    encoder_width = 64
    decoder_width = 64

    decoder_depth = 3
    encoder_depth = 3

    # latent dim
    ns = [8, 16, 32]

    kl_weight = 0.5

    save_folder = fstr("../vano_results/{problem}/")

    for n in ns:
        project = "vano"
        model_name = "vano"
        save_file_name = fstr(save_folder.payload + "{project}_n={n}_seed={seed}.json")
        param_save_file_name = fstr(save_folder.payload + "{project}_n={n}_seed={seed}.pickle")
        log_str = fstr("VANO on {problem} with {opt_choice} n={n} seed={seed}")

        encoder_config = dict(
            activation = activation_choice,
            last_layer_activate = True,
            num_hidden_layers = encoder_depth,
            nodes = encoder_width,
            output_dim = n * 2,
            name = "encoder"
        )

        decoder_x_config = dict(
            activation = activation_choice,
            num_hidden_layers = decoder_depth,
            nodes = decoder_width,
            output_dim = 64,
            name = "decoder_x"
        )

        decoder_z_config = dict(
            activation = activation_choice,
            num_hidden_layers = decoder_depth,
            nodes = decoder_width,
            output_dim = 64,
            name = "decoder_z"
        )

        decoder_joint_network_config = dict(
            activation = activation_choice,
            num_hidden_layers = decoder_depth+1,
            nodes = decoder_width,
            output_dim = m,
            name = "decoder_joint_network"
        )

        model_config = dict(
            decoder_config = dict(
                decoder_x_config = decoder_x_config,
                decoder_z_config = decoder_z_config,
                decoder_joint_network_config = decoder_joint_network_config
            ),
            encoder_config = encoder_config,
            latent_dim = n,
            key = jax.random.PRNGKey(seed)
        )

        if not os.path.isdir(str(save_folder)):
            os.makedirs(str(save_folder))

        print("\n\n")
        print("+"*100)
        print(log_str)
        print("+"*100)
        print("\n\n")

        train_input, Y, test_input, Y_test = get_data()

        dummy_input = (jnp.expand_dims(train_input[0][0, :], 0), jnp.expand_dims(train_input[1][0, :], 0))

        hyperparameter_dict = dict(
            print_bool = print_bool,
            print_interval = print_interval,
            epochs = epochs,
            model_config = model_config,
            opt_choice = opt_choice,
            schedule_choice = schedule_choice,
            lr_dict = lr_dict,
            problem = problem,
            N = N,
            N_test = N_test,
            n = n,
            m = m,
            dtype = dtype,
            seed = seed
        )

        train_config = dict(
            dummy_input = dummy_input,
            train_input = train_input,
            test_input = test_input,
            Y = Y,
            Y_test = Y_test,
            model_name = model_name,
        ) | hyperparameter_dict

        logged_results, trained_params = train_func(train_config)

        # removing array keys
        del hyperparameter_dict["model_config"]["key"]

        logged_results = logged_results | hyperparameter_dict

        if save_results_bool:
            # torch.save(trained_params, str(param_save_file_name))
            save_results(logged_results, str(save_file_name))

