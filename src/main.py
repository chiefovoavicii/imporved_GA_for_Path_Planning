'''
主程序入口
'''

import sys
import os


from pathlib import Path

current_file = Path(__file__).resolve()
# 项目根目录假设为当前文件的上两级目录（根据实际结构调整）
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.ui.config_parser import config
from src.core.environment.grid import GridEnvironment
from src.core.environment.obstacles import ObstacleManager
from src.core.environment.poc.generate_poc import GeneratePOC
from core.algorithms import genetic
from core.utils.visualize import Visualize
from src.ui.gui import launch_gui
from src.ui.gui.param_panel import ParamPanel_env, ParamPanel_algo
from src.ui.config_parser import load_config
from src.ui.gui.interface import MainInterface

import tkinter as tk
from tkinter import ttk

def main():



    def on_ui_close():

        print(config)

        #interface = MainInterface(config = config)

        #interface.mainloop()


        # 类实例化
        env = GridEnvironment(config)
        gp = GeneratePOC(env)
        obs = ObstacleManager(config)
        vis = Visualize()

        gp.initialize_poc()
        obs.initialize_obstacles()
        obs.ensure_start_pos_clear(config['num_plane'] + config['num_ship'])  # 确保起始位置没有障碍物


  
        # 遗传算法
        if config["algorithm"] == 'genetic':
            trained_agent = genetic.train_ga(config, env, gp, obs)
            path = genetic.generate_ga_path(trained_agent)
       
        # 可视化cdp和reward变化图
        #vis.visualize_cdp_and_reward(cdp_list, reward_list)

        # 可视化路径
        vis.visualize(env, obs, gp, path, config)

    interface = MainInterface(config, on_ui_close)
    interface.mainloop()  # 这里会阻塞直到窗口关闭


if __name__ == "__main__":
        main()

