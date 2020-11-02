import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import progressbar
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def visualize_predictions(args, sess, net, replay_memory, env, e=0):
    """Plot predictions for a system against true time evolution
    Args:
        args: Various arguments and specifications
        sess: TensorFlow session
        net: Neural network dynamics model
        replay_memory: Object containing training/validation data
        env: Simulation environment
        e: Current training epoch
    """
	# Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
    x[:] = replay_memory.x_test

    # Find number of times to feed in input
    n_passes = 200//args.batch_size

    # Initialize array to hold predictions
    preds = np.zeros((1, 2*args.seq_length-1, args.state_dim))
    for t in range(n_passes):
        # Construct inputs for network
        feed_in = {}
        feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
        feed_out = [net.A, net.B, net.z_vals, net.x_pred_reshape_unnorm]
        out = sess.run(feed_out, feed_in)
        A = out[0]
        B = out[1]
        z1 = out[2].reshape(args.batch_size, args.seq_length, args.latent_dim)[:, -1]
        x_pred = out[3]
        x_pred = x_pred[:, :-1]

        preds = np.concatenate((preds, x_pred), axis=0)       
    preds = preds[1:]

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm((preds[:, :args.seq_length] - sess.run(net.shift))/sess.run(net.scale) - x[0, :args.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)
        
    # Plot different quantities
    x = x*sess.run(net.scale) + sess.run(net.shift)

    # # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    plt.close()
    f, axs = plt.subplots(args.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for i in range(args.state_dim):
        axs[i].plot(range(1, 2*args.seq_length), x[0, :-1, i], 'k')
        axs[i].plot(range(1, 2*args.seq_length), preds[ind0, :, i], 'r')
        axs[i].plot(range(1, 2*args.seq_length), preds[ind1, :, i], 'g')
        axs[i].plot(range(1, 2*args.seq_length), pred_mean[:, i], 'b')
        axs[i].fill_between(range(1, 2*args.seq_length), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        axs[i].set_ylim([np.amin(x[0, :, i])-0.2, np.amax(x[0, :, i]) + 0.2])
    plt.xlabel('Time Step')
    plt.xlim([1, 2*args.seq_length-1])
    plt.savefig('vk_predictions/predictions_' + str(e) + '.png')
 
def pendulum_cost(states, us, gamma):
    """Define cost function for inverted pendulum
    Args:
        states: Sequence of state values [num_models, N, state_dim]
        us: Sequence of control inputs [N-1, action_dim]
        gamma: Discount factor
    Returns:
        List of (discounted) cost values at each time step
    """
    num_models = len(states)
    N = states.shape[1]
    thetas = np.arctan2(states[:, :, 1], states[:, :, 0])

    # Find cost of states alone, averaged across models
    cost = np.square(thetas) + 0.1*np.square(states[:, :, 2])
    cost[:, :-1] += 0.001*np.square(np.sum(us, axis=1))
    exp_cost = np.mean(cost, axis=0)

    # Return discounted cost
    return [gamma**t*exp_cost[t] for t in range(N)]
