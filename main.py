import json
import configparser
import os
from openai import OpenAI

from typing import TypeVar, Generator

#from filter_papers import filter_by_gpt
from Filter import filter_paper
from parse_json_to_md import render_md_string

T = TypeVar("T")

def batched(items: list[T], batch_size: int) -> list[T]:
    # takes a list and returns a list of list with batch_size
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

def translate_to_chinese_via_deepseek(text: str, client: OpenAI,config) -> str:
    """
    使用 DeepSeek API 将英文文本翻译成中文
    """
    try:
        # 使用 DeepSeek API 的 chat.completions.create 方法
        response = client.chat.completions.create(
            model=config["SELECTION"]["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates English text to Chinese."},
                {"role": "user", "content": f"Translate the following text to Chinese:\n\n{text}"},
            ],
            stream=False,
            temperature=1.0,
            seed=0
        )
        # 获取返回的翻译文本
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        return text



if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    OAI_KEY = "sk-823fc37d344844128a4bbd02e690d198"

    '''if "deepseek" in config["SELECTION"]["model"]:
        openai_client = OpenAI(api_key=OAI_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError()'''
    openai_client = OpenAI(
        api_key="1e2aba84196b4e3694a51fc3d30d7a0f.m1XJ5Bcncw8aWOuc",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    with open('paper_list/ICLR_2025_1.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)

    paper_list = {}
    for paper in papers:
        paper_list[paper["id"]] = paper
    selected_papers = filter_paper(
        paper_list,
        config,
        openai_client,
    )

    print(selected_papers)
    #增加翻译成中文的模块
    # 对筛选出的论文标题和摘要进行翻译
    for paper in selected_papers:
        print(paper)
        print(f"Translating paper: {paper['title']}")
        paper['title_cn'] = translate_to_chinese_via_deepseek(paper['title'], openai_client,config)
        paper['abstract_cn'] = translate_to_chinese_via_deepseek(paper['abstract'], openai_client,config)


    if config["OUTPUT"].getboolean("dump_json"):
        with open(config["OUTPUT"]["output_path"] + "output.json", "w") as outfile:
            json.dump(selected_papers, outfile, indent=4)
    if config["OUTPUT"].getboolean("dump_md"):
        with open(config["OUTPUT"]["output_path"] + "output.md", "w") as f:
            f.write(render_md_string(selected_papers))
        # 生成包含中文翻译的 Markdown 文件
        with open(config["OUTPUT"]["output_path"] + "output_translated.md", "w") as f:
            for paper in selected_papers:
                f.write(f"## {paper['title_cn']}\n\n")
                f.write(f"{paper['abstract_cn']}\n\n")

