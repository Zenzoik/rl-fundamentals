import numpy as np
from rl.algorithms.common.td_agent import TemporalDifferenceAgent
from rl.common.results_logger import ResultsLogger


class QLearning(TemporalDifferenceAgent):
    """
    Q-Learning algorithm for Temporal Difference learning.

    Args:
        env: The environment to interact with.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration parameter for epsilon-greedy policy.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(
        self,
        env,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        logger: ResultsLogger = None,
        random_seed: int = None
    ) -> None:
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name: str = "Q-Learning"

    def learn(self, num_episodes: int = 500) -> None:
        """
        Trains the Q-Learning agent for a given number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        # HOMEWORK BEGINS: Implement the Q-learning algorithm (~14 lines). Refer to the Sarsa implementation.
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # target uses the best next action, not the one we actually take
                td_target = reward + self.gamma * np.max(self.q_values.get(next_state))
                td_error = td_target - self.q_values.get(state, action)
                self.q_values.update(state, action, self.q_values.get(state, action) + self.alpha * td_error)

                state = next_state
                self.logger.log_timestep(reward)
                done = terminated or truncated

            self.logger.log_episode()
        # HOMEWORK ENDS
