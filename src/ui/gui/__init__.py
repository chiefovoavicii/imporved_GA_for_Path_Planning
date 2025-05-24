from .interface import MainInterface

def launch_gui(config):
    """启动GUI入口函数"""
    app = MainInterface(config)
    app.mainloop()