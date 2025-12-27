import infrastructure.utils.torch_utils as tu
from infrastructure.utils.logger import Logger

import gymnasium as gym
import numpy as np

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

NAME = "Jarda Dortomil :P"
UCOS = [514286, 514324]


class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state: int, *args, **kwargs):
        raise NotImplementedError()

    # Should return the predicted logits for the given state
    def raw(self, state: int, *args, **kwargs):
        raise NotImplementedError()

    # Should return the predicted value of the given state V(state)
    def value(self, state: int, *args, **kwargs):
        raise NotImplementedError()


class Trainer:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma: float, steps: int, *args, **kwargs) -> Policy:
        raise NotImplementedError()


class ValueNet(nn.Module):
    """Critic Network for PPO."""
    def __init__(self, input_size, output_size=1, hidden_size=64):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def value_no_grad(self, obs):
        return self(obs)

    def value(self, obs):
        return self(obs)


class PolicyNet(nn.Module):
    """Actor Network for PPO."""

    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def play(self, obs):
        output = self(obs)
        dist = Categorical(logits=output)
        action = dist.sample()
        return action.item()

    def log_probs(self, obs, actions):
        output = self(obs)
        dist = Categorical(logits=output)
        return dist.log_prob(actions)

    @torch.no_grad()
    def log_probs_no_grad(self, obs, actions):
        self.log_probs(obs, actions)

    @torch.no_grad()
    def entropy(self, obs):
        output = self(obs)
        dist = Categorical(logits=output)
        return dist.entropy()


class PPOPolicy(Policy):
    def __init__(self, net: PolicyNet, value_net: ValueNet):
        self.net = net
        self.value_net = value_net

    # Returns played action in state
    def play(self, state):
        return self.net.play(state)

    # Returns value
    def value(self, state):
        return self.value_net.value_no_grad(state)


def collect_trajectories(env, policy, step_limit, gamma, bootstrap_trunc):
    """
    This is a helper function that collects a batch of episodes,
    totalling `step_limit` in steps. The last episode is truncated to
    accomodate for the given limit.


    You can use this during training to get the necessary data for learning.

        Returns several flattened tensors:

            1) States encountered
            2) Actions played
            3) Rewards collected
            4) Dones - Points of termination / truncation.


        Whenever done[i] is True, then (states[i], actions[i], rewards[i]) is
        the last valid transition of the episode. The data on index i+1 describe
        the first transition in the following episode.

        If `bootstrap_trunc` is true and an episode is truncated at timestep i,
        gamma * policy.value(next_state) is added to rewards[i]. Note that if you
        are not utilizing a critic network, this should be turned off.

    You can modify this function as you see fit or even replace it entirely.

    """

    states, actions, rewards, dones, acc_rews = [], [], [], [], []
    steps = 0
    acc_rew = 0

    while steps < step_limit:

        obs, _ = env.reset()
        obs = tu.to_torch(obs)
        obs = obs.float()

        done = False
        while not done:
            action = policy.play(obs)

            states.append(obs)
            actions.append(action)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            steps += 1
            next_obs = tu.to_torch(next_obs)

            truncated = truncated or steps == step_limit

            # Optionally bootstrap on truncation
            if truncated and bootstrap_trunc:
                bootstrap = tu.to_numpy(gamma * policy.value_no_grad(obs))[0]
                reward += bootstrap

            rewards.append(reward)
            acc_rew += reward

            if terminated or truncated:
                done = True
                acc_rews.append(acc_rew)
                acc_rew = 0

            dones.append(done)
            obs = next_obs
    
    return states, actions, rewards, dones, acc_rews


class PPOTrainer(Trainer):
    def __init__(self, env, state_dim, num_actions,
                 policy_lr=1e-3, value_lr=1e-3, gae_lambda=0.99, batch_size=128, hidden_size=64, verbose=False,
                 clipp_epsilon=0.2, coll_traj_steps=10000, num_batches=60, lr_decay=0.95, entropy_coef=0.01):
        """
            env: The environment to train on
            state_dim: The dimension of the state space
            num_actions: The number of actions in the action space
            policy_lr: The learning rate for the policy network.
            value_lr: The learning rate for the value network.
            gae_lambda: The GAE discounting parameter lambda
            batch_size: The batch size (num of steps from env for each
            learning iteration)
        """

        self.env = env
        self.batch_size = batch_size

        self.policy_net = PolicyNet(state_dim, num_actions, hidden_size)
        self.old_policy_net = PolicyNet(state_dim, num_actions, hidden_size)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.old_policy_net.eval()
        self.value_net = ValueNet(state_dim, 1)

        # Optimizers for each of the nets
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                 lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                lr=value_lr)

        self.gae_lambda = gae_lambda
        self.clip_epsilon = clipp_epsilon
        self.coll_traj_steps = coll_traj_steps
        self.num_batches = num_batches
        self.policy_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=num_batches, gamma=lr_decay)
        self.value_scheduler = lr_scheduler.StepLR(self.value_optimizer, step_size=num_batches, gamma=lr_decay)
        self.entropy_coef = entropy_coef
        self.verbose = verbose

    def train(self, gamma, train_steps) -> PPOPolicy:
        """
            Train the agent for number of steps specified by `train_steps`,
            while using the supplied discount `gamma`.

            Training will proceed by sampling batches of episodes
            using `collect_trajectories` and constructing the appropriate
            loss function.
        """

        learning_steps = train_steps // self.coll_traj_steps
        self.env.reset()

        for i in range(learning_steps):
            policy = PPOPolicy(self.policy_net, self.value_net)
            
            # Collect trajectories
            states, actions, rewards, dones, acc_rews = collect_trajectories(self.env, policy, self.coll_traj_steps, gamma,
                                                                             bootstrap_trunc=False)
            
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
 
            # batches = self.minibatch_indices(states, actions, rewards, dones)
            batches = self.minibatch_indices(states, n_batches=self.num_batches)
           
            # Logging average episode rewards
            if self.verbose:
                avg_reward = sum(acc_rews[:-1]) / max(len(acc_rews) - 1, 1)
                print(f"Iteration {i}, Average Reward: {avg_reward:.2f}, "
                      f"lr policy: {self.policy_optimizer.param_groups[0]['lr']:.5f}, "
                      f"lr value: {self.value_optimizer.param_groups[0]['lr']:.5f}")

            state_tensor = torch.stack(states)
            action_tensor = torch.tensor(actions)    
                
            returns = self.calculate_returns(rewards, dones, gamma)
            advantages = self.calculate_gae(rewards, state_tensor, dones, gamma)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for minibatch in batches:
                # Prepare tensors for training
                states_mini = state_tensor[minibatch]
                actions_mini = action_tensor[minibatch]

                old_logprobs = self.old_policy_net.log_probs(states_mini, actions_mini).detach()

                self.update(states_mini, actions_mini, advantages[minibatch], 
                            returns[minibatch], old_logprobs)
        
        return PPOPolicy(self.policy_net, self.value_net)

    def calculate_returns(self, rewards, dones, gamma):

        """
            For each collected timestep in the environment, calculate the
            discounted return from that point to the end of episode
        """

        res = torch.zeros(len(rewards))
        discounted_return = 0.0

        for i in range(len(rewards) - 1, -1, -1):

            if dones[i]:
                discounted_return = 0.0

            discounted_return = rewards[i] + gamma * discounted_return
            res[i] = discounted_return

        return res

    def calculate_gae(self, rewards, states, dones, gamma):
        """
            For each collected timestep in the environment, calculate the
            Generalized Advantage Estimate.
        """

        res = torch.zeros(len(rewards))

        values = self.value_net.value_no_grad(states)
        gae = 0.0

        for i in range(len(rewards) - 1, -1, -1):
            next_value = 0 if dones[i] else values[i + 1]
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * self.gae_lambda * gae * (1 - dones[i])
            res[i] = gae + values[i]

        return res

    def ppo_loss(self, old_logprobs, logprobs, advantages):
        ratios = torch.exp(logprobs - old_logprobs)
        clipped_loss = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        return -torch.min(ratios * advantages, clipped_loss).mean()

    def value_loss(self, values, returns):
        values = values.squeeze(-1)
        return torch.nn.functional.mse_loss(values, returns)

    def minibatch_indices(self, states, n_batches):
        return np.random.randint(0, len(states), (n_batches, self.batch_size))

    def update(self, states, actions, advantages, returns, old_logprobs):
        """
        Update the policy and value networks using PPO.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions taken.
            advantages (torch.Tensor): Estimated advantages.
            returns (torch.Tensor): Discounted returns for each state.
            old_logprobs (torch.Tensor): Log probabilities under the old policy.
        """
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        values = self.value_net(states)
        logprobs = self.policy_net.log_probs(states, actions)
        entropy = self.policy_net.entropy(states)

        policy_loss = self.ppo_loss(old_logprobs, logprobs, advantages)
        value_loss = self.value_loss(values, returns)

        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()
        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.policy_scheduler.step()
        self.value_scheduler.step()


def get_env_dimensions(env):
    """
        Helper function to get dimensions of state/action spaces of gym environments.
    """

    def get_space_dimensions(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        else:
            raise TypeError(
                f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

    state_dim = get_space_dimensions(env.observation_space)
    num_actions = get_space_dimensions(env.action_space)

    return state_dim, num_actions


def train_cartpole(env, train_steps, gamma) -> PPOPolicy:
    num_inputs, num_outputs = get_env_dimensions(env)
    trainer = PPOTrainer(env, num_inputs, num_outputs, batch_size=64, policy_lr=0.005, value_lr=0.005,
                         entropy_coef=0.001, gae_lambda=0.95, clipp_epsilon=0.15,
                         coll_traj_steps=5000, num_batches=75, lr_decay=0.9, verbose=False)
    return trainer.train(gamma, train_steps)


def train_acrobot(env, train_steps, gamma) -> PPOPolicy:
    num_inputs, num_outputs = get_env_dimensions(env)
    trainer = PPOTrainer(env, num_inputs, num_outputs, batch_size=64, policy_lr=0.005,
                         value_lr=0.005, entropy_coef=0.01, gae_lambda=0.95, clipp_epsilon=0.2,
                         coll_traj_steps=5000, num_batches=100, lr_decay=0.9)
    return trainer.train(gamma, train_steps)


def train_lunarlander(env, train_steps, gamma) -> PPOPolicy:
    num_inputs, num_outputs = get_env_dimensions(env)
    trainer = PPOTrainer(env, num_inputs, num_outputs, batch_size=128, policy_lr=0.005, verbose=True,
                         value_lr=0.005, hidden_size=32, gae_lambda=0.95, clipp_epsilon=0.2,
                         coll_traj_steps=10000, num_batches=100, lr_decay=0.9, entropy_coef=0.01)
    return trainer.train(gamma, train_steps)


"""
    CarRacing is a challenging environment for you to try to solve.
"""

RACING_CONTINUOUS = False


def train_carracing(env, train_steps, gamma) -> PPOPolicy:
    """
        As the observations are 96x96 RGB images you can either use a
        convolutional neural network, or you have to flatten the observations.

        You can use gymnasium wrappers to achieve the second goal:
    """
    env = gym.wrappers.FlattenObservation(env)

    states, num_actions = get_env_dimensions(env)
    trainer = PPOTrainer(env, states, num_actions, batch_size=128, policy_lr=0.003, value_lr=0.003,
                         gae_lambda=0.95, clipp_epsilon=0.2, coll_traj_steps=5000, num_batches=60,
                         lr_decay=0.95, entropy_coef=0.01, verbose=False)
    return trainer.train(gamma, train_steps)


def test_policy(policy, env, num_episodes=10):
    """
    Tests the given policy on the CartPole environment.

    Args:
        policy: The trained policy to be tested.
        env: The CartPole environment instance.
        num_episodes: Number of test episodes to run.

    Returns:
        rewards: A list of total rewards gained in each episode.
    """
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy.play(obs_tensor)  # Get the action from the policy
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    return rewards, np.mean(rewards)


if __name__ == "__main__":
    """
        The flag RACING_CONTINUOUS determines whether the CarRacing environment
        should use a continuous action space. Set it to True if you want to
        experiment with a continuous action space. The evaluation will be done
        based on the value of this flag.
    """
    #CARTRACING
    # env = gym.make("CarRacing-v3", continuous=RACING_CONTINUOUS)
    # policy = train_carracing(env, 100000, 0.99)
    # _, avg_rew = test_policy(policy, env, 10)
    # print(f"Average reward: {avg_rew}")


    # CARTPOLE
    # env = gym.make("CartPole-v1")
    # policy = train_cartpole(env, 100000, 0.99)
    # _, avg_rew = test_policy(policy, env, 10)
    # print(f"Average reward: {avg_rew}")

    # #ACROBOT
    # env = gym.make("Acrobot-v1")
    # policy = train_acrobot(env, 100000, 0.99)
    # _, avg_rew = test_policy(policy, env, 10)
    # print(f"Average reward: {avg_rew}")

    #LUNARLANDER
    env = gym.make("LunarLander-v3")
    policy = train_lunarlander(env, 300000, 0.99)
    _, avg_rew = test_policy(policy, env, 10)
    print(f"Average reward: {avg_rew}")

