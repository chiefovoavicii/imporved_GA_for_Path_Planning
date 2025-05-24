import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import os, sys
current_file = Path(__file__).resolve()
# 项目根目录假设为当前文件的上两级目录（根据实际结构调整）
root_dir = current_file.parent.parent.parent.parent
sys.path.append(str(root_dir))


from src.ui.gui.param_panel import ParamPanel_env, ParamPanel_algo
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

class MainInterface(tk.Tk):
    def __init__(self, config, on_config_ready):
        super().__init__()
        #self.wm_attributes('-transparentcolor', '#ab23ff')
        #style = ttk.Style()
        #style.configure('Transparent.TFrame', background='#ab23ff')

        self.title("搜索路径规划配置系统")
        self.geometry("600x400")

        #图片背景
        self.bg_image = None  # 保持引用防止被回收
        self.bg_canvas = None
        # 主框架背景色匹配透明色

        self.config = config  # 保持对主config的引用
        self.on_config_ready = on_config_ready

        # 创建主布局框架
        self.main_frame = ttk.Frame(self.bg_image)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建三大功能区
        self._create_algorithm_selector()
        self._create_param_panel(config)
        self._create_control_buttons()
        

    def _create_algorithm_selector(self):
        """算法选择区域"""
        algo_frame = ttk.LabelFrame(self.main_frame, text="算法选择",style='Transparent.TFrame')
        algo_frame.pack(fill=tk.BOTH, padx=10, pady=10) #框的位置
        
        self.algo_var = tk.StringVar()
        algorithms = [('强化学习', 'rl'), ('遗传算法', 'genetic'), ('贪婪算法', 'greedy')]
        
        for text, value in algorithms:
            rb = ttk.Radiobutton(algo_frame, text=text, value=value, variable=self.algo_var)
            rb.pack(side=tk.LEFT, padx=15)
        
    def _create_param_panel(self, config):
        """参数配置面板"""
        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 环境配置标签页
        env_frame = ttk.Frame(notebook)
        self.param_panel = ParamPanel_env(env_frame, config)
        notebook.add(env_frame, text="环境参数")
        
        # 算法参数标签页（可根据算法动态加载）
        self.algo_frame = ttk.Frame(notebook)
        self.algo_panel = ParamPanel_algo(self.algo_frame, config)
        notebook.add(self.algo_frame, text="算法参数")
        
    def _create_control_buttons(self):
        """操作按钮区域"""
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="保存配置", command=self._save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="启动算法", command=self._run_algorithm).pack(side=tk.RIGHT)

    def _save_config(self):
        """更新配置到主字典"""
        # 获取参数面板的值
        new_config_env = self.param_panel.get_values()  # 环境参数
        new_config_algo = self.algo_panel.get_values()  # 算法参数
        
        # 合并更新配置（算法参数优先级高于环境参数）
        self.config.update(new_config_env)
        self.config.update(new_config_algo)
        # 更新主配置字典
        self.config['algorithm'] = self.algo_var.get()
        print("更新后的配置:", self.config)

    def _run_algorithm(self):
        """执行算法并关闭界面"""
        self._save_config()
        self.destroy()  # 关闭UI窗口
        self.on_config_ready()  # 触发回调执行后续逻辑


def convert_tuple_list(value):
        """将字符串格式的坐标列表转换为Python元组列表"""
        try:
        # 去除所有空格并规范格式
            cleaned = value.replace(" ", "").replace("),(", ")|(").replace("],[", ")|[").strip("[]")
            # 分割独立坐标单元
            tuples = []
            for item in cleaned.split("|"):
                if not item: continue
                # 解析单个坐标
                coord = item.strip("()")
                x, y = map(int, coord.split(","))
                tuples.append((x, y))
            return tuples
        except Exception as e:
            raise ValueError(f"格式转换错误: {value}") from e
            
'''
    def _create_background(self):
        
        
        """创建背景画布"""
        try:
            
            # 创建背景画布
            self.bg_canvas = tk.Canvas(self, highlightthickness=0)
            self.bg_canvas.pack(fill="both", expand=True)

            
            pil_image = Image.open("C:/Users/Jianze/search-path-planning/pictures/R-C.png")
            self.bg_image = ImageTk.PhotoImage(pil_image)
            self.bg_canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

            # 将原有主框架放在背景之上
            self.main_frame = ttk.Frame(self.bg_canvas)
            self.bg_canvas.create_window(50, 50, window=self.main_frame, anchor="nw")
            
        except ImportError:
            print("警告：未安装PIL库，使用纯色背景")
            self.configure(background="#F0F0F0")
        except Exception as e:
            print(f"加载背景图片失败: {str(e)}")

    def _create_result_panel(self):
        """右侧结果展示面板"""
        result_frame = ttk.LabelFrame(self.main_frame, text="算法可视化", width=500, height=500)
        result_frame.pack_propagate(0)  # 固定尺寸
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建绘图容器 
        self.plot_canvas = tk.Canvas(result_frame, bg='white', width=450, height=450)
        self.plot_canvas.pack(pady=10)
        
        # 初始化空图像占位符
        self.plot_canvas.create_rectangle(0, 0, 450, 450, outline="#CCCCCC", dash=(4,2))
        self.plot_canvas.create_text(225, 225, text="等待算法运行结果...", fill="gray")
        
        # 初始化Matplotlib画布 
        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_frame)
        self.canvas.get_tk_widget().pack_forget()  # 初始隐藏

    def _update_plot(self, data):
        """更新可视化图表"""
        # 清除旧内容
        self.ax.clear()
        self.plot_canvas.delete("all")
        
        # 绘制新图表（示例）
        x = [1,2,3,4,5]
        y = [2,3,5,7,11]
        self.ax.plot(x, y, 'r-')
        self.ax.set_title("路径规划结果")
        
        # 显示Matplotlib画布 
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()
'''