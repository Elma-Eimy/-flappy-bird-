import argparse
import os.path
from email import iterators

import matplotlib
import torch
from time import sleep
from FlappyCNN import FlappyCNN
from Flappy_bird import FlappyBird
from Process_image import process_image
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="图片的长宽")
    parser.add_argument("--saved_path", type=str, default="./train_result/models", help='模型的存放路径')

    args = parser.parse_args()
    return args


def test(opt, i, game_num, game_sore):
    torch.manual_seed(520)
    model = torch.load(f=os.path.join(opt.saved_path, f'flappy_bird_{i}00000'), map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = process_image(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        if reward == -1 or game_sore > 350:
            game_num += 1
            if game_num == 5:
                return game_sore // 5
        game_sore += reward
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        # print(time)
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = process_image(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        state = next_state


if __name__ == "__main__":
    opt = get_args()
    iteration = []
    game_Sore = []
    # 进行测试的模型，可用自己修改范围内测试对应的模型，例如0~27就是测试0到260万次的模型
    # 如果是单独测试某个模型等会可用设置对应的范围i来进行测试
    for i in range(26, 27):
        game_num = 0
        game_sore = 0
        game_sore = test(opt, i, game_num, game_sore)
        iteration.append(i * 100000)
        game_Sore.append(game_sore)
        print("迭代", i * 100000, " 奖励", game_sore)
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(20, 8), dpi=80)
    plt.ylabel('平均得分')
    plt.xlabel('迭代次数')
    plt.plot(iteration, game_Sore)
    plt.savefig("iteration-game_Sore.jpg")

