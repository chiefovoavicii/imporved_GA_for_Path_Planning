import tkinter as tk
from tkinter import ttk,messagebox
 
#智能体数量，障碍物参数，
class ParamPanel_env(ttk.Frame):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.vars = {}
        self.dist_frame_obs = None
        self.dist_frame_poc = None   
        self.poc_mapping = [
        ('动态高斯分布', 'dynamic_gaussian'),
        ('高斯分布', 'gaussian'),
        ('零一分布', '01'),
        ('均匀分布', 'uniform'),
        ('均匀分布', 'uniform')
    ]
        # 创建两列布局
        left_frame = ttk.Frame(self)
        right_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self._create_basic_params(left_frame)
        self._create_distribution_params(right_frame)
        self.pack(fill=tk.BOTH, expand=True)
        

    def _create_input_row(self, parent, label, key, row):
        """通用输入行创建方法"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        
        if isinstance(self.config[key], bool):
            var = tk.BooleanVar(value=self.config[key])
            cb = ttk.Checkbutton(parent, variable=var)
            cb.grid(row=row, column=1)
        else:
            var = tk.StringVar(value=str(self.config[key]))
            entry = ttk.Entry(parent, textvariable=var, width=15)
            entry.grid(row=row, column=1)
        
        self.vars[key] = var
        return row + 1
    
    def _create_basic_params(self, parent):
        """基础参数区域"""
        frame = ttk.LabelFrame(parent, text="基础配置")
        frame.pack(pady=5, fill=tk.X)
        
        
        ttk.Label(frame, text="障碍物类型").grid(row=0, column=0, sticky=tk.W)
        obs_var = tk.StringVar(value=self.config['obstacle_type'])
        cb = ttk.Combobox(frame, textvariable=obs_var, 
                        values=['fixed', 'random'])
        cb.grid(row=0, column=1)
        self.vars['obstacle_type'] = obs_var

        self.dist_frame_obs = ttk.Frame(frame)
        self.dist_frame_obs.grid(row=1, columnspan=2, sticky=tk.W)
        self._update_dist_params(obs_var.get())

        obs_var.trace_add('write', lambda *_: self._update_dist_params(obs_var.get()))
        self._update_dist_params(obs_var.get())
        
    def _create_distribution_params(self, parent):
        """分布参数区域"""
        frame = ttk.LabelFrame(parent, text="目标分布配置")
        frame.pack(pady=5, fill=tk.X)
        
        # 分布类型选择
        ttk.Label(frame, text="分布类型").grid(row=0, column=0, sticky=tk.W)
        poc_var = tk.StringVar(value=self.config['poc_type'])
        cb = ttk.Combobox(frame, textvariable=poc_var, 
                        values=['dynamic_gaussian','gaussian', '01', 'uniform'])
        cb.grid(row=0, column=1)
        self.vars['poc_type'] = poc_var
        # 初始化概率分布框架
        self.dist_frame_poc = ttk.Frame(frame)
        self.dist_frame_poc.grid(row=1, columnspan=2, sticky=tk.W)
    
    # 绑定事件
        poc_var.trace_add('write', lambda *_: self._update_dist_params(poc_var.get()))
        self._update_dist_params(poc_var.get())  # 初始更新
    
    def _update_dist_params(self, dist_type):
        """更新分布参数界面"""
        # 分离处理障碍物和概率分布
        if dist_type in ['fixed', 'random']:
            # 处理障碍物参数
            if self.dist_frame_obs is None:
                self.dist_frame_obs = ttk.Frame(self.master)
                self.dist_frame_obs.grid(row=1, columnspan=2, sticky=tk.W)
            
            for widget in self.dist_frame_obs.winfo_children():
                widget.destroy()

            params = [('obstacle_density', '随机障碍物密度'),
                      ('grid_size', '网格尺寸（整数）'),
                    ('pod', 'POD值（0-1）'), 
                     ] if dist_type == 'random' else [
                      ('grid_size', '网格尺寸（整数）'),
                    ('pod', 'POD值（0-1）'), ]
            
            row = 0
            for key, label in params:
                row = self._create_input_row(self.dist_frame_obs, label, key, row)
        else:
            # 处理概率分布参数
            if self.dist_frame_poc is None:
                self.dist_frame_poc = ttk.Frame(self.master)
                self.dist_frame_poc.grid(row=1, columnspan=2, sticky=tk.W)
            
            for widget in self.dist_frame_poc.winfo_children():
                widget.destroy()

            row = 0
            if dist_type == 'gaussian':
                params = [
                    ('gauss_center', '中心坐标（x,y）'),
                    ('sigma', '标准差（浮点数）'),
                    ('theta', '阈值（0-1）')
                ]
            elif dist_type == '01':
                params = [('target_pos_01', '目标位置（x,y）')]
            else:
                params = []
            
            for key, label in params:
                row = self._create_input_row(self.dist_frame_poc, label, key, row)
    


    def get_values(self):
        config = {}
        for key, var in self.vars.items():
            value = var.get()
            try:
                if key in ['grid_size', 'max_steps', 'episodes']:
                    config[key] = int(value)
                elif key in ['pod', 'sigma', 'theta']:
                    config[key] = float(value)
                elif key in ['gauss_center', 'target_pos_01']:
                    config[key] = tuple(map(int, value.strip('()').split(',')))
                else:
                    config[key] = value
            except ValueError as e:
                raise ValueError(f"参数{key}格式错误: {str(e)}")
        return config
    

class ParamPanel_algo(ttk.Frame):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.vars = {}
           
        
        # 创建两列布局
        left_frame = ttk.Frame(self)
        right_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self._create_basic_params(left_frame)
        self._create_distribution_params(right_frame)
        self.pack(fill=tk.BOTH, expand=True)
        


    def _create_input_row(self, parent, label, key, row):
        """通用输入行创建方法"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        
        if isinstance(self.config[key], bool):
            var = tk.BooleanVar(value=self.config[key])
            cb = ttk.Checkbutton(parent, variable=var)
            cb.grid(row=row, column=1)
        else:
            var = tk.StringVar(value=str(self.config[key]))
            entry = ttk.Entry(parent, textvariable=var, width=15)
            entry.grid(row=row, column=1)
        
        self.vars[key] = var
        return row + 1
    
    def _create_basic_params(self, parent):
        """基础参数区域"""
        frame = ttk.LabelFrame(parent, text="训练参数")
        frame.pack(pady=5, fill=tk.X)
        
        row = 0
        params = [
            ('max_steps', '最大步数'),
            ('episodes', '训练回合数'),
        ]
        
        for key, label in params:
            row = self._create_input_row(frame, label, key, row)
            
    def _create_distribution_params(self, parent):
        """分布参数区域"""
        frame = ttk.LabelFrame(parent, text="搜索平台参数")
        frame.pack(pady=5, fill=tk.X)
        
        # 分布类型选择
        row = 0
        params = [
            ('num_plane','飞机平台位置'),
            ('num_ship','舰船平台位置')
        ]
        
        for key, label in params:
            row = self._create_input_row(frame, label, key, row)
        



    def get_values(self):
        config = {}
        for key, var in self.vars.items():
            value = var.get()
            try:
                if key in ['grid_size', 'max_steps', 'episodes']:
                    config[key] = int(value)
                elif key in ['num_plane', 'num_ship']:
                    config[key] = convert_tuple_list(value)
                elif key in ['pod', 'sigma', 'theta']:
                    config[key] = float(value)
                elif key in ['gauss_center', 'target_pos_01']:
                    config[key] = tuple(map(int, value.strip('()').split(',')))
                else:
                    config[key] = value
            except ValueError as e:
                raise ValueError(f"参数{key}格式错误: {str(e)}")
        return config
    
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