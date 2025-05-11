from itertools import cycle
from time import sleep
from numpy.random import randint
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np


class FlappyBird(object):
    init()

    # 记录游戏的时间
    fps_clock = time.Clock()  # 控制运行速度
    screen_width = 288  # 游戏窗口宽度
    screen_height = 512   # 游戏窗口高度
    screen = display.set_mode((screen_width, screen_height))  # 设置游戏窗口宽高
    display.set_caption('强化学习Flappy Bird')  # 设置标题
    base_image = load('assets\\sprites\\base.png').convert_alpha()  # 设置地表图片
    background_image = load('assets\\sprites\\background-black.png').convert()  # 加载背景图片
    pipe_images = [rotate(load('assets\\sprites\\pipe-green.png').convert_alpha(), 180),
                   load('assets\\sprites\\pipe-green.png').convert_alpha()]  # 设置管道的图片，一个设置为倒立的，另外一个不是
    bird_images = [load('assets\\sprites\\redbird-downflap.png').convert_alpha(),
                   load('assets\\sprites\\redbird-midflap.png').convert_alpha(),
                   load('assets\\sprites\\redbird-upflap.png').convert_alpha()]  # 分别设置小鸟运动的图片

    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]  # 设置小鸟的碰撞体箱
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]  # 设置管道的碰撞体箱

    # 每秒传输帧数
    fps = 30  # 设置游戏帧数，即每秒播放图片的数量
    pipe_gap_size = 100     # 设置每个管道的间距
    pipe_velocity_x = -4    # 设置每个管道的前行速度，横向速度，因为管道不会上下飘动

    # 鸟的元素
    min_velocity_y = -8     # 鸟的最小纵向速度，负数即坠落速度
    max_velocity_y = 10     # 鸟的最大纵向速度，整数即上升速度
    downward_speed = 1      # 鸟的向下速度
    upward_speed = -9       # 鸟的向上速度

    bird_index_generator = cycle([0, 1, 2, 1])      # 生成鸟的索引生成器

    def __init__(self):
        # 初始化小鸟、管子
        self.iter = self.bird_index = self.score = 0    # 设置管道的迭代，鸟的索引，以及分数

        self.bird_width = self.bird_images[0].get_width()   # 获取鸟的宽度
        self.bird_height = self.bird_images[0].get_height()     # 获取鸟的高度
        self.pipe_width = self.pipe_images[0].get_width()       # 获取管道的宽度
        self.pipe_height = self.pipe_images[0].get_height()     # 获取的管道的高度

        self.bird_x = int(self.screen_width/5)      # 一直设置鸟的横坐标为窗口靠左的五分之一位置
        self.bird_y = int((self.screen_height-self.bird_height)/2)  # 初始设置鸟的纵坐标为窗口的正中心

        # 地的初始函数
        self.base_x = 0     # 设置地面的 横坐标位置
        self.base_y = self.screen_height * 0.79     # 设置地表的纵坐标，因为一般的窗口大小是以左上角为原点
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()   # 设置地表的适应宽度

        pipes = [self.generate_pipe(), self.generate_pipe()]    # 获取一两组管道的数据
        pipes[0]['x_upper'] = pipes[0]['x_lower'] = self.screen_width   # 设置第一组上下管道的横坐标
        pipes[1]['x_upper'] = pipes[1]['x_lower'] = self.screen_width * 1.5  # 设置第二组上下管道的横坐标
        self.pipes = pipes

        self.current_velocity_y = 0     # 初始化当前小鸟的速度
        self.is_flapped = False     # 看是否在飞翔，即是否起跳

    def generate_pipe(self):
        x = self.screen_width + 10      # 设置生成位置
        gap_y = randint(2, 10)*10+int(self.base_y/5)    # 开始随机生成上下管道之间的间隙
        return {"x_upper": x, 'y_upper': gap_y-self.pipe_height, 'x_lower': x, "y_lower": gap_y + self.pipe_gap_size}   #  返回上下管道的信息

    def is_collided(self):
        # 检查鸟是否触地
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)   # 生成小鸟的碰撞箱
        pipe_boxes = []     # 生成管道的碰撞箱
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe['x_upper'], pipe['y_upper'], self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe['x_lower'], pipe['y_lower'], self.pipe_width, self.pipe_height))
            # 检擦鸟的边框有没有和管道的边框重叠
            if bird_bbox.collidelist(pipe_boxes) == -1:     # 先简单地检查小鸟和管道图片是否有重叠部分
                return False
            for i in range(2):      # 开始检验小鸟和管道的像素碰撞
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])    # 获取小鸟和管道的交集部分
                min_x1 = cropped_bbox.x - bird_bbox.x   # 开始获取相对位置
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1:min_x1+cropped_bbox.width,
                          min_y1:min_y1+cropped_bbox.height]*self.pipe_hitmask[i][min_x2:min_x2+cropped_bbox.width,
                          min_y2:min_y2+cropped_bbox.height]):      # 检验是否发生了碰撞
                    return True
            return False

    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False
        # 检查输入动作
        if action == 1:
            self.current_velocity_y = self.upward_speed     # 设置飞翔速度，即起跳
            self.is_flapped = True      # 飞翔吧~

        # 更新分数
        bird_center_x = self.bird_x + self.bird_width/2
        for pipe in self.pipes:
            pipe_center_x = pipe['x_upper']+self.pipe_width/2
            if pipe_center_x < bird_center_x < pipe_center_x+5:
                self.score += 1
                reward = 1
                break

        # 更新 index and iteration
        if (self.iter+1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x+100) % self.base_shift)
        # 更新鸟的位置
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(self.current_velocity_y, self.bird_y-self.current_velocity_y-self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        # 更新管道位置
        for pipe in self.pipes:
            pipe['x_upper'] += self.pipe_velocity_x  # 前进吧管道
            pipe['x_lower'] += self.pipe_velocity_x

        # 更新管道
        if 0 < self.pipes[0]['x_lower'] < 5:
            self.pipes.append(self.generate_pipe())     # 添加更多的管道
        if self.pipes[0]['x_lower'] < -self.pipe_width:
            del self.pipes[0]
        if self.is_collided():  # 如果碰撞了
            terminal = True
            reward = -1
            self.__init__()     # 重新进行游戏

        self.screen.blit(self.background_image, (0, 0))    # 显示游戏图片
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        image = array3d(display.get_surface())  # 将游戏画面转为np数组
        display.update()
        self.fps_clock.tick(self.fps)
        return image, reward, terminal
