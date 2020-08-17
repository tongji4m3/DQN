from DQN.my_env import Env
from DQN.RL_brain import SumDQN
from DQN.hdqn.h_dqn import HDqnAgent
import tensorflow as tf
import numpy as np
def run_maze():
    step = 0#用来控制什么时候开始训练
    reward_sum=0#每十回合统计的奖励
    step_sum=0#每十回合统计的步数


    for episode in range(1001):
        # initial observation
        observation = env.reset()
        external_step_counter=1
        external_reward=0
        #人流量随机改变
        #env.resetPeople()

        done = False
        if (episode % 10 == 0):
            env.resetPeople()
        while not done:
            #中层循环（选子目标）,写一个get_subgoal,传入参数当前位置observation，返回一个子目标的坐标

            internal_step_counter = 1

            observation0=observation
            internal_reward=0
            goal_reach=False

            #done和goal_reach作为底层循环的while条件
            #while not done and not goal_reach：
            while True:
                # RL choose action based on observation
                #choose_action要根据subgoal和当前位置一起选，这个函数要改
                action = HDqnAgent.choose_action(observation)
                step+=1
                # RL take action and get next observation and reward,（这里应该返回internal_reward
                observation_, reward, done = env.step(action)

                #判断是否到达子目标
                # goal_reach,reward_sum = HDqnAgent.check_get_subgoal(observation,reward_sum)

                #if goal_reach:
                   #加入一个到达子目标的奖励，增加内部奖励internal_reward+=...

                #注意要开两个sumtree，把metacontroller的经验和controller的经验分开来存
                #存经验要改：要加入subgoal作为参数，表示把智能体想到达这个subgoal而做的行动
                HDqnAgent.store_transition(observation, action, reward, observation_)

                #同时训练高层和底层
                if (step > 1000) and (step % 5 == 0):
                    HDqnAgent.learn()
                    #加入一个meta_learn,同时训练两个controller

                # swap observation
                observation = observation_

                #定义两个step_counter，分别用于到达子目标的步数计数和外部的计数
                env.update_env(episode, internal_step_counter,done,reward)

                #定义一个external_reward来接收每一次内部循环产生的internal_reward，用于外部的训练，（external_reward+=internal_reward）
                #external_reward += internal_reward
                internal_step_counter += 1

                if goal_reach:
                    external_step_counter += internal_step_counter
                    break

            # 在中层循环写一个if done，用来跳出中层循环
            if done:
                break



            #在外部再调用一次存储记忆，设置一个外部奖励，用于meta_controller训练（hdqn.store_transition())
            #外层经验参数为（observation0,sub_goal,external_reward,_observation)
            # print(reward_sum)
        if (episode%10==0) &(episode!=0):
            env.Reward_memory(episode/10,reward_sum/10,step_sum/10)
            reward_sum=0
            step_sum=0
    # end of game
    env.draw()
    print('game over')


if __name__ == "__main__":
    # maze game
    env = Env()
    HDqnAgent=HDqnAgent()
    run_maze()


    #RL.save("my_net/save_net.ckpt")
    #graph = tf.get_default_graph()
    # saver=tf.train.Saver()
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     save_path=saver.save(sess,"my_net/save_net.ckpt")
    #     print("Save to path:",save_path)



