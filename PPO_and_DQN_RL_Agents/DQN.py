from matplotlib import pyplot as plt

from infrastructure.utils.logger import Logger
import infrastructure.utils.torch_utils as tu

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from torch.optim.lr_scheduler import ExponentialLR

import random



"""
    The Policy/Trainer interface remains the same as in the first assignment:
"""

class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, *args, **kwargs) -> int:
        raise NotImplementedError()

    # Should return the predicted Q-values for the given state
    def raw(self, state: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class Trainer:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma : float, steps : int, *args, **kwargs) -> Policy:
        raise NotImplementedError()

class ReplayBuffer:
    def __init__(self, size, n_steps, gamma):
        self.buffer = deque(maxlen=size)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


"""
    The goal in the second assignment is to implement your own DQN agent, along with
    some additional features. The mandatory ones include:

    1) Target network for bootstrapping
    2) Double DQN
    3) N-step returns for calculating the target
    4) Scheduling of the epsilon parameter over time
    
    
    DISCLAIMER:
    
    All the provided code is just a template that can help you get started and 
    is not mandatory to use. You only need to stick to the interface and the
    method signatures of the constructor and `train` for DQNTrainer.

    Some of the extensions above can be implemented in multiple ways - 
    like exponential averaging vs hard updates for the target net.
    You can choose either, or even experiment with both.
"""


class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

    @torch.no_grad()
    def play(self, obs, eps=0.0):

        qvals = self(obs)
        if np.random.rand() <= eps:
            return np.random.choice(len(qvals))

        # You can also randomly break ties here.
        x = torch.argmax(qvals)

        # Cast from tensor to int so gym does not complain
        return int(x)


class DQNPolicy(Policy):
    def __init__(self, net : DQNNet):
        self.net = net

    def play(self, state):
        return self.net.play(state)

    def raw(self, state: int) -> torch.Tensor:
        return self.net(state)


class DQNTrainer(Trainer):
    DQN = "DQN"
    DQN_TARGET = "DQN+target"
    DOUBLE_DQN = "DoubleDQN"

    def __init__(
            self, env, state_dim, num_actions,
            lr=0.004, mini_batch=128, max_buffer_size=8000, n_steps=1,
            initial_eps=0.25, final_eps=0.01, mode=DQN_TARGET, target_update_interval=100,
            **kwargs
        ) -> None:
        super(DQNTrainer, self).__init__(env)
        """
            Initialize the DQNTrainer
            
            Args:
                env: The environment to train on
                state_dim: The dimension of the state space
                num_actions: The number of actions in the action space
                lr: The learning rate
                mini_batch: The mini batch size
                max_buffer_size: The maximum replay buffer size
                n_steps: The number of steps to look ahead when calculating targets
                initial_eps: The initial epsilon value for epsilon-greedy exploration
                final_eps: The final epsilon value for epsilon-greedy exploration
                mode: The mode of operation. Can be "DQN", "DQN+target", "DoubleDQN"
        """
        self.device = self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # CHANGE if using server with one gpu
        self.net = DQNNet(state_dim, num_actions).to(self.device)         # main network
        self.target_net = DQNNet(state_dim, num_actions).to(self.device)  # target network
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-3)         # optimizer
        self.buffer = ReplayBuffer(max_buffer_size, n_steps, gamma=0.99)  # replay buffer
        self.mini_batch = mini_batch                                      # mini batch size
        self.n_steps = n_steps                                            # n-step bootstrapping
        self.initial_eps = initial_eps                                    # initial epsilon for epsilon-greedy
        self.final_eps = final_eps                                        # final epsilon for epsilon-greedy
        self.target_update_interval = target_update_interval              # target network update interval
        self.gamma = 0.99                                                 # discount factor
        self.mode = mode                                                  # mode of operation
        self.num_actions = num_actions                                    # number of actions in the action space
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.9)      # learning rate scheduler
        self.step = 0                                                     # current step

    def loss_fn(self, qvals, target_qvals):
        return nn.MSELoss()(qvals, target_qvals)

    def calculate_targets(self, transition_batch):
        states, actions, rewards, next_states, dones = transition_batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.net(states).gather(1, actions).squeeze()

        if self.mode == self.DQN:
            next_q_values = self.net(next_states).max(1)[0]
        elif self.mode == self.DQN_TARGET:
            next_q_values = self.target_net(next_states).max(1)[0]
        elif self.mode == self.DOUBLE_DQN:
            best_actions = self.net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # n-step bootstrapping
        targets = rewards + (self.gamma ** self.n_steps) * next_q_values * (1 - dones)
        return q_values, targets

    def update_net(self, *args):
        batch = args[0]

        qvals, target_qvals = self.calculate_targets(batch)

        loss = self.loss_fn(qvals, target_qvals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.step % 5000 == 0:
            self.lr_scheduler.step()

    def linear_decay(self, step, total_steps):
        return max(self.final_eps, self.initial_eps - (self.initial_eps - self.final_eps) * (step / total_steps))

    def exponential_decay(self, step, total_steps):
        dec_rate = (self.final_eps / self.initial_eps) ** (1 / total_steps)
        return max(self.final_eps, self.initial_eps * (dec_rate ** step))

    def train(self, gamma, train_time_steps, logger=None, verbose=None): #-> DQNPolicy:
        init_state, _ = self.env.reset()
        step = 0
        acc_rew = 0                 # accumulated reward during training
        acc_disc_rew = 0            # accumulated discounted reward during training
        num_episodes = 0            # number of episodes during training
        rewards = []                # episodic rewards
        disc_rewards = []           # episodic discounted rewards
        num_epsiodes_finished = 0   # number of episodes that finished in the last 1000 steps
        disc_rew = (np.ones_like(self.n_steps)*gamma)**np.arange(0, self.n_steps)
        n_step_buffer = deque(maxlen=self.n_steps)

        while step < train_time_steps:
            cur_disc = 1
            epp_rew = 0
            epp_disc_rew = 0
            done = False
            num_episodes += 1
            state = self.env.reset()[0]

            while not done and step < train_time_steps:
                # exponential decay
                eps = self.exponential_decay(step, train_time_steps)

                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.net.play(state_t, eps)
                succ, rew, terminated, truncated, _ = self.env.step(action)
                n_step_buffer.append(rew)
                acc_rew += rew
                epp_disc_rew += cur_disc * rew
                acc_disc_rew += cur_disc * rew
                cur_disc *= gamma
                epp_rew += rew

                """
                    TODO: 
                        1) Save the transition into the replay buffer.
                        2) Sample a minibatch from the buffer
                        3) Update the main network
                        4) (Possibly) update the target network as well.
                """
                if terminated or truncated:
                    done = True
                    rewards.append(epp_rew)
                    disc_rewards.append(epp_disc_rew)
                    num_epsiodes_finished += 1

                if len(n_step_buffer) >= self.n_steps:
                    rew = disc_rew @ np.array(n_step_buffer)
                    transition = (state, action, rew, succ, int(done))
                    self.buffer.add(transition)

                if len(self.buffer) >= self.mini_batch:
                    batch = self.buffer.sample(batch_size=self.mini_batch)
                    self.update_net(batch)

                # update target network every target_update_interval steps
                if step % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                if step % 1000 == 0:
                    # average reward and discounted reward of epsiodes that finished in the last 1000 steps
                    mov_rew = sum(rewards[-num_epsiodes_finished:]) / num_epsiodes_finished if num_epsiodes_finished > 0 else 0
                    mov_disc_rew = sum(disc_rewards[-num_epsiodes_finished:]) / num_epsiodes_finished if num_epsiodes_finished > 0 else 0
                    if logger is not None:
                        init_state_action_values = self.net(tu.to_torch(init_state).to(self.device))
                        logger.write(
                            {
                                "reward": acc_rew,
                                "num_episodes": num_episodes,
                                "avg_reward": acc_rew / num_episodes,
                                "avg_discounted_reward": acc_disc_rew / num_episodes,
                                "max_init_state_action_value": torch.max(init_state_action_values).item(),
                                "avg_init_state_action_value": torch.mean(init_state_action_values).item(),
                                "mov_rew": mov_rew,
                                "mov_disc_rew": mov_disc_rew,

                            },
                            step,
                        )
                    num_epsiodes_finished = 0

                if step % 2000 == 0 and verbose:
                    print(
                        f"Step: {step}, Epsilon: {eps:.3f},"
                        f" Avg Reward: {acc_rew / num_episodes:.3f},"
                        f" Avg Discounted Reward: {acc_disc_rew / num_episodes:.3f},"
                        f" lr: {self.lr_scheduler.get_last_lr()[0]:.3f}")

                state = succ
                step += 1
                self.step += 1
                
        if verbose == 'plot':
            window_size = 10
            adjusted_average_rewards = [
                np.mean(rewards[i:i + window_size]) for i in range(len(rewards) - window_size + 1)
            ]
    
            # Generate x values for the adjusted average rewards
            adjusted_episodes = np.arange(len(adjusted_average_rewards))
    
            # Plotting the adjusted moving average
            plt.figure(figsize=(10, 6))
            plt.plot(adjusted_episodes, adjusted_average_rewards, label="Adjusted 10-Episode Moving Average", color="blue")
            plt.scatter(range(len(rewards)), rewards, label="Rewards Per Episode", color="orange", alpha=0.6)
            plt.title(f"mode: {self.mode}")
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            plt.show()

        return DQNPolicy(self.net), rewards


"""
    Helper function to get dimensions of state/action spaces of gym environments.
"""
def get_env_dimensions(env):

    def get_space_dimensions(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return space.shape[0]
        else:
            raise TypeError(f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

    state_dim = get_space_dimensions(env.observation_space)
    num_actions = get_space_dimensions(env.action_space)

    return state_dim, num_actions


"""
    Demonstration code - get states/actions, play randomly
"""
def example_human_eval(env_name):
    env = gym.make(env_name)
    state_dim, num_actions = get_env_dimensions(env)

    trainer = DQNTrainer(env, state_dim, num_actions)
    # Tensor operations example
    trainer.calculate_targets([])

    # Train the agent on 1000 steps.
    pol = trainer.train(0.99, 1000)

    # Visualize the policy for 10 episodes
    human_env = gym.make(env_name, render_mode="human")
    for _ in range(10):
        state = human_env.reset()[0]
        done = False
        while not done:
            action = pol.play(tu.to_torch(state))
            state, _, done, _, _ = human_env.step(action)


if __name__ == "__main__":
    env_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
    env = gym.make(env_names[0])
    state_dim, num_actions = get_env_dimensions(env)

    logger = Logger("logs/try1/")

    DQN = "DQN"
    DQN_TARGET = "DQN+target"
    DOUBLE_DQN = "DoubleDQN"

    trainer = DQNTrainer(env, state_dim, num_actions)
    trainer.train(0.99, 50000, logger=logger, verbose='plot')




