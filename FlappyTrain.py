import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import FlappyCNN
import Flappy_bird
import Process_image

import matplotlib.pyplot as plt


def get_args():
    # 可以使用命令行输入参数
    # 设置一下训练的必要参数
    parser = argparse.ArgumentParser("""A test of Q-learning to play flappy bird""")
    parser.add_argument('--image_size', type=int, default=84, help='所有数据样本的公共宽高,默认宽高84*84')
    parser.add_argument('--batch_size', type=int, default=32, help='每批数据的图像数，默认32张一批')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='选择不同的优化算法')
    parser.add_argument('--lr', type=int, default=1e-6, help='设置学习率，建议选择范围为1e-4~1e-6')
    parser.add_argument('--gamma', type=float, default=0.99, help='设置未来奖励的折现值')
    parser.add_argument('--initial_epsilon', type=float, default=0.1, help="""初始的探索策略概率，
    也就是说模型会以多少的概率做出随机动作以获取经验，因为一开始的模型并没有多少经验所以我们可以设置一个较大的贪心概率""")
    parser.add_argument('--final_epsilon', type=float, default=1e-4, help="""最后收敛后做出随机动作的概率，
    也就是说当前模型已经学习到了足够的经验，做出鲁莽的行动的概率应该很小，但是为了保证模型的活性，仍然需要设置一个较小的随机动作概率""")
    parser.add_argument('--num_iters', type=int, default=50000, help='设置迭代训练次数')
    parser.add_argument('--replay_memory_size', type=int, default=30000, help='设置智能体和环境互动的经验内存大小')
    parser.add_argument('--log_path', type=str, default='train_result\\tensorboard', help="设置训练日志存放路径")
    parser.add_argument('--save-path', type=str, default='train_result', help='设置模型存放路径')
    args = parser.parse_args()
    return args


def train(opt):
    # 检验GPU资源是否可用，并设置随机数种子
    if torch.cuda.is_available():
        print(1)
        torch.cuda.manual_seed(520)
    else:
        torch.manual_seed(520)
    model = FlappyCNN.FlappyCNN()   # 引入卷积模型
    # model = torch.load("flappy_bird".format(opt.saved_path), map_location=lambda storage, loc:storage)
    if os.path.isdir(opt.log_path):     # 检验训练日志路径，并更新训练日志，以免发生冲突
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    log_writer = SummaryWriter(opt.log_path)    # 创立日志记录对象
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)     # 读取学习效率参数，使用Adam优化方法，自适应学习效率，而且在训练大量数据时有较快的收敛速度
    criterion = nn.MSELoss()    # 使用MSE（均方差函数）这一对损失值比较敏感的函数以最小化损失
    game_state = Flappy_bird.FlappyBird()   # 获取游戏状态
    image, reward, terminal = game_state.next_frame(0)  # 先获取初始小鸟活动的信息
    image = Process_image.process_image(image[:game_state.screen_width, :int(game_state.base_y)],
                                        opt.image_size, opt.image_size)     # 输入图像信息，进行预处理工作，可以刨除地面的大部分显示，以减少训练所需要的空间
    image = torch.from_numpy(image)     # 将图像转为torch的张量
    if torch.cuda.is_available():       # 检验gpu是否可用，将卷积模型放在gpu中加速处理
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]    # 初始时四张图片合并在一起以清晰小鸟的运动轨迹，并额外的添加维度，满足Pytorch的卷积需求

    replay_memory = []      # 建立样本池子，以供深度学习进行训练
    """
    loss_memory = []
    iter_num = []
    """
    iter = 0        # 迭代次数
    while iter < opt.num_iters:     # 如果迭代次数小于给定的次数则循环
        prediction = model(state)[0]        # 返回预期值
        # 建立基础的贪心策略算法
        epsilon = opt.final_epsilon + ((opt.num_iters - iter)*(opt.initial_epsilon-opt.final_epsilon)/opt.num_iters)
        u = random()    # 随机产生动作
        random_action = u <= epsilon
        if random_action:
            print("随机产生一个动作")  # 产生的动作随着iter增大而减少
            action = randint(0, 1)      # 探索中
        else:
            action = torch.argmax(prediction).item()    # 开始利用策略产生动作，即使用期望最大的值来产生动作
        next_image, reward, terminal = game_state.next_frame(action)    # 获取产生动作后的信息
        # 对产生后的动作的信息进行处理
        next_image = Process_image.process_image(next_image[:game_state.screen_width, :int(game_state.base_y)],
                                                 opt.image_size, opt.image_size)
        # 数组转为PyTorch特有的数据张量，对张量的修改也会影响到原始的数组数据
        next_image = torch.from_numpy(next_image)

        # 在给定的维度上对输入的张量进行连续操作
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]     # 踢掉第一帧的图片，组成新的待卷积数据
        replay_memory.append([state, action, reward, next_state, terminal])     # 将数据加进样本池中
        if len(replay_memory) > opt.replay_memory_size:  # 如果发现池子满了，则踢掉第一个样本，持续更新
            del replay_memory[0]

        # 从样本池子replay_memory中随机抽取一定数量的样本，以列表的形式返回
        # 每次抽取就是一次迭代训练
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)  # 拆解数据
        # 连续操作
        state_batch = torch.cat(tuple(state for state in state_batch))  # 合并数据，作为一批
        # 数组转换为张量，使用将动作值的0，1转为独热编码，方便分类处理
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32)
        )
        # 数组转为张量
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        # 分别使用GPU加速
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)       # 生成预期值束
        next_prediction_batch = model(next_state_batch)  # 生成下一个预期束
        # 连续操作y_batch张量，即计算目标值,实现贝尔曼方程
        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction
                  in zip(reward_batch, terminal_batch, next_prediction_batch)))

        # 当前q_value张量
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()  # 用的梯度包含上一个batch的，相当于batch_size为之前的两倍，所以optimizer.step()是用在batch里的
        loss = criterion(q_value, y_batch)
        loss.backward()  # 根据网络反向传播的梯度信息来更新网络参数
        optimizer.step()  # 更新学习
        # 状态更新
        state = next_state
        iter += 1
        print(iter)
        '''
        # 状态监测
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                    iter + 1,
                    opt.num_iters,
                    action,
                    loss,
                    epsilon, reward, torch.max(prediction)))'''
        # 记录生成日志
        log_writer.add_scalar('Train/Loss', loss, iter)
        log_writer.add_scalar('Train/Epsilon', float(epsilon), iter)
        log_writer.add_scalar('Train/Reward', reward, iter)
        log_writer.add_scalar('Train/Q-value', torch.max(prediction).item(), iter)
        ''' # 进行损失函数的采样
                if(iter+1) % 500 == 0:# 迭代500次采样一次loss
                    loss_memory.append(loss.item())
                    iter_num.append(iter+1)'''
        if (iter + 1) % 50000 == 0:  # 训练到一定的程度就进行储存
            print(iter + 1)
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))
        '''
        # 生成折线图可选
            if (iter+1) % 1000000 == 0:#共采样2000个点
                plt.figure(figsize=(20, 8), dpi=80)
                plt.ylabel('Recon_loss')
                plt.xlabel('iter_num')
                print(iter_num,loss_memory)
                lt.plot(iter_num,loss_memory)
                plt.savefig("{}/flappy_bird_{}.jpg".format(opt.saved_path, iter+1))
                    '''
    torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
