from DQN.my_env import Env
from DQN.Double_DQN.Double_RL_brain import DoubleDQN


def run_maze():
    step = 0
    for episode in range(501):
        # initial observation
        observation = env.reset()

        step_counter = 1
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            env.update_env(episode, step_counter,done,reward)

            if done:
                break
            step += 1
            step_counter+=1
        if (episode%20==0) &(episode!=0):
            env.Reward_memory(episode)
    # end of game
    env.draw()
    print('game over')


if __name__ == "__main__":
    # maze game
    env = Env()
    RL = DoubleDQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      # output_graph=True
                      )
    run_maze()

