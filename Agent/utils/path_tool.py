'''
为整个工程提供统一绝对路径
'''

import os

def get_project_root() -> str:
    '''
    获取项目根目录
    '''
    # 获取当前py文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取工程的根目录 文件所在文件夹绝对路径 -> 工程根目录
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path: str) -> str:
    '''
    给相对路径返回绝对路径
    获取绝对路径
    '''
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path


if __name__ == '__main__':

    print(get_abs_path("data"))