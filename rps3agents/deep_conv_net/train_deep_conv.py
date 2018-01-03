import random

import matplotlib.pyplot as plt
import gym
import numpy as np
# noinspection PyUnresolvedReferences
import rps3env
import tensorflow as tf

OBS_SHAPE = [None, 28, 1, 3]
NUM_ACTIONS = 28 * 28

LEARN_RATE = 0.001
DECAY_RATE = 0.95

NUM_EPISODES = 100


def deep_conv_net(input_layer):
    with tf.name_scope('conv'):
        conv1 = tf.layers.conv2d(
            inputs=input_layer, filters=9, kernel_size=[5, 1], padding='same', activation=tf.nn.relu, name='conv1'
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[3, 1], strides=3, name='pool1'
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1, filters=18, kernel_size=[3, 1], padding='same', activation=tf.nn.relu, name='conv2'
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 1], strides=1, name='pool2'
        )
        pool2_flat = tf.layers.flatten(pool2, name='conv_flat')
    with tf.name_scope('dense') as s:
        dense1 = tf.contrib.layers.fully_connected(
            inputs=pool2_flat, num_outputs=84, activation_fn=tf.nn.relu, scope=s
        )
    with tf.name_scope('output') as s:
        output = tf.contrib.layers.fully_connected(
            inputs=dense1, num_outputs=NUM_ACTIONS, activation_fn=tf.nn.relu, scope=s
        )
    return output


def get_observation(obs):
    return np.concatenate([
        np.array(obs['occupied'], dtype=np.float32).reshape([28, 1, 1]),
        np.array(obs['player_owned'], dtype=np.float32).reshape([28, 1, 1]),
        np.array(obs['piece_type'], dtype=np.float32).reshape([28, 1, 1]),
    ], axis=2)


def get_available_actions(available_actions):
    action_indices = np.ravel_multi_index(tuple(zip(*available_actions)), (28, 28))
    action_filter = np.eye(NUM_ACTIONS)[action_indices].sum(axis=0)
    return action_filter


def main():
    tf.reset_default_graph()

    state_input = tf.placeholder(tf.float32, OBS_SHAPE, name='state_input')
    expected_output = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='expected_output')
    available_actions = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='available_actions')

    with tf.name_scope("deep_conv_net"):
        nn_output = deep_conv_net(state_input)
        # tf.summary.histogram('nn_output', nn_output)
        filtered_output = tf.multiply(nn_output, available_actions, name='filtered_output')

        best_action = tf.argmax(filtered_output, axis=1, name='best_action')

        loss_fn = tf.losses.mean_squared_error(expected_output, nn_output)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
        train_op = optimizer.minimize(loss=loss_fn, global_step=tf.train.get_global_step())

    episode_rewards = tf.placeholder(tf.float32, (None, 1), name='episode_rewards')
    tf.summary.scalar('mean_episode_reward', tf.reduce_mean(episode_rewards))
    tf.summary.scalar('last_episode_reward', episode_rewards[-1, 0])
    tf.summary.histogram('episode_reward_values', episode_rewards)

    # output graph for tensorboard
    summary_writer = tf.summary.FileWriter('graph')
    summary_writer.add_graph(tf.get_default_graph())

    # merge summary operators
    merged_summaries = tf.summary.merge_all()

    # create environment
    env = gym.make('RPS3Game-v0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        exploration_prob = 0.1
        episode_reward_values = []

        for i in range(NUM_EPISODES):
            # initialize the environment consistently every time
            env.seed(0)
            env.reset()
            obs, reward, done, info = env.step([1, 2, 3] * 3)

            total_episode_reward = 0

            while not done:
                # env.render()
                obs_extracted = get_observation(obs)

                a_best, q_values = sess.run([best_action, filtered_output], feed_dict={
                    state_input: np.array([obs_extracted]),
                    available_actions: np.array([get_available_actions(env.available_actions)]),
                })
                if np.random.rand(1) < exploration_prob or not q_values.max():
                    a_best[0] = np.ravel_multi_index(random.choice(env.available_actions), (28, 28))

                action = np.unravel_index(a_best[0], (28, 28))
                assert action in env.available_actions

                obs, reward, done, info = env.step(action)
                move_reward = sum(reward)
                total_episode_reward += move_reward

                new_q_values = sess.run(nn_output, feed_dict={
                    state_input: np.array([get_observation(obs)])
                })

                target_q_values = q_values
                target_q_values[0, a_best[0]] = move_reward + DECAY_RATE * np.max(new_q_values)

                _ = sess.run(train_op, feed_dict={
                    state_input: np.array([obs_extracted]),
                    expected_output: target_q_values
                })

            env.render()
            print('Finished episode {} with total reward {}'.format(i, total_episode_reward))

            episode_reward_values.append(total_episode_reward)

            # reduce exploration probability gradually
            exploration_prob = 1. / ((i / 50) + 10)

            summary = sess.run(merged_summaries, feed_dict={
                episode_rewards: np.array(episode_reward_values).reshape((-1, 1))
            })
            summary_writer.add_summary(summary, i)

        env.close()


if __name__ == '__main__':
    main()
