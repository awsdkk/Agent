from .logger_handler import logger
from .config_handler import prompt_config
from .path_tool import get_abs_path

def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompt_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"prompt_config.yml中未找到main_prompt_path")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"读取system_prompt_path失败: {str(e)}")
        raise e

def load_rag_prompt():
    try:
        rag_summarize_prompt_path = get_abs_path(prompt_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"prompt_config.yml中未找到rag_summarize_prompt_path")
        raise e

    try:
        return open(rag_summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"读取rag_summarize_prompt_path失败: {str(e)}")
        raise e

def load_report_prompt():
    try:
        report_prompt_path = get_abs_path(prompt_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"prompt_config.yml中未找到report_prompt_path")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"读取report_prompt_path失败: {str(e)}")
        raise e

if __name__ == "__main__":
    print(load_system_prompt())
    print(load_rag_prompt())
    print(load_report_prompt())