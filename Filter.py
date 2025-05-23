import dataclasses
import json
import re

import retry
import openai

print(openai.__version__)
from tqdm import tqdm


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

@retry.retry(tries=3, delay=2)
def call_chatgpt(full_prompt, openai_client, model):
    return openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0,
        seed=0
    )


def run_and_parse_chatgpt(full_prompt, openai_client, config):
    # just runs the chatgpt prompt, tries to parse the resulting JSON
    completion = call_chatgpt(full_prompt, openai_client, config["SELECTION"]["model"])
    out_text = completion.choices[0].message.content
    out_text = re.sub("```jsonl\n", "", out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()
    # split out_text line by line and parse each as a json.
    json_dicts = []
    for line in out_text.split("\n"):
        # try catch block to attempt to parse json
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                print(completion.choices[0].message.content)
            continue
    return json_dicts

def paper_to_string(paper_entry) -> str:
    # renders each paper into a string to be processed by GPT
    new_str = (
            "ID: "
            + paper_entry['id']
            + "\n"
            + "Title: "
            + paper_entry['title']
            + "\n"
            + "Authors: "
            + " and ".join(paper_entry['authors'])
            + "\n"
            + "Abstract: "
            + paper_entry['abstract'][:4000]
    )
    return new_str


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


def run_on_batch(
        paper_batch, base_prompt, criterion, postfix_prompt, openai_client, config
):
    batch_str = [paper_to_string(paper) for paper in paper_batch]
    full_prompt = "\n".join(
        [
            base_prompt,
            criterion + "\n",
            "\n\n".join(batch_str) + "\n",
            postfix_prompt,
        ]
    )
    json_dicts = run_and_parse_chatgpt(full_prompt, openai_client, config)
    return json_dicts

def paper_to_titles(paper_entry) -> str:
    return "ID: " + paper_entry["id"] + " Title: " + paper_entry["title"] + "\n"

def filter_by_title(
    paper_dicts, config, openai_client, base_prompt, criterion
):
    papers = list(paper_dicts.values())
    filter_postfix = 'Identify any papers that are absolutely and completely irrelevant to the criteria, formatted as a list of ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. '
    batches_of_papers = batched(papers, int(config["SELECTION"]["batch_size"]))
    final_list = []
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)
        out_text = completion.choices[0].message.content
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                final_list.append(paper)
                if paper["id"] not in filtered_set:
                    final_list.append(paper)
                else:
                    print("Filtered out paper " + paper.title) # 删掉的title
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(completion)
            continue
    selected_papers = {}
    for paper in final_list:
        selected_papers[paper["id"]] = paper
    return selected_papers


def filter_by_abstract(paper_dicts, config, openai_client, base_prompt, criterion, postfix_prompt):
    selected_papers_list = []
    paper_list = list(paper_dicts.values())
    batch_of_papers = batched(paper_list, int(config["SELECTION"]["batch_size"]))
    for batch in tqdm(batch_of_papers):
        json_dicts = run_on_batch(
            batch, base_prompt, criterion, postfix_prompt, openai_client, config
        )
        #print("摘要筛选")
        #print(json_dicts)
        for jdict in json_dicts:
            if (
                    bool(jdict["RELEVANCE"])
                    and jdict["ID"] in paper_dicts
            ):
                selected_papers_list.append(paper_dicts[jdict["ID"]])
    print(
        str(len(selected_papers_list))
        + " papers after title and abs filtering"
    )
    return selected_papers_list

def filter_paper(paper_list, config, openai_client):
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()

    selected_papers_dicts = filter_by_title(
        paper_list, config, openai_client, base_prompt, criterion
    )
    #print("题目筛选")
    #print(selected_papers_dicts)
    selected_papers_list = filter_by_abstract(
        selected_papers_dicts, config, openai_client, base_prompt, criterion, postfix_prompt
    )
    return selected_papers_list