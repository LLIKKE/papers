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


def calc_price(model, usage):
    if model == "gpt-4-1106-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    elif model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    elif (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0
    elif model == "deepseek-chat":
        return ((0.014 * usage.prompt_tokens) / 1_000_000) + ((0.28 * usage.completion_tokens) / 1_000_000)
    elif model == "glm-4-flash":
        return 0
    else:
        return 0


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
    return json_dicts, calc_price(config["SELECTION"]["model"], completion.usage)


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
    #print(full_prompt)
    json_dicts, cost = run_and_parse_chatgpt(full_prompt, openai_client, config)
    return json_dicts, cost

def paper_to_titles(paper_entry) -> str:
    return "ID: " + paper_entry["id"] + " Title: " + paper_entry["title"] + "\n"

def filter_papers_by_title(
    papers, config, openai_client, base_prompt, criterion
):
    filter_postfix = 'Identify any papers that are absolutely and completely irrelevant to the criteria, formatted as a list of ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. '
    batches_of_papers = batched(papers, 1)
    final_list = []
    cost = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        #print(full_prompt)
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)
        cost += 0
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
    return final_list, cost


def filter_by_gpt(
        paper_list, config, openai_client, all_papers, selected_papers
):
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    all_cost = 0
    if config["SELECTION"].getboolean("run_openai"):

        paper_list, cost = filter_papers_by_title(
            paper_list, config, openai_client, base_prompt, criterion
        )
        print(paper_list)
        # batch the remaining papers and invoke GPT
        batch_of_papers = batched(paper_list, int(config["SELECTION"]["batch_size"]))
        scored_batches = []
        for batch in tqdm(batch_of_papers):
            scored_in_batch = []
            json_dicts, cost = run_on_batch(
                batch, base_prompt, criterion, postfix_prompt, openai_client, config
            )
            all_cost += cost
            for jdict in json_dicts:
                jdict["RELEVANCE"] = True
                print(jdict)
                if (
                        bool(jdict["RELEVANCE"])
                        and jdict["ID"] in all_papers
                ):
                    selected_papers[jdict["ID"]] = all_papers[jdict["ID"]]
                scored_in_batch.append(all_papers[jdict["ID"]])
            scored_batches.append(scored_in_batch)
        if config["OUTPUT"].getboolean("dump_debug_file"):
            with open(
                    config["OUTPUT"]["output_path"] + "gpt_paper_batches.debug.json", "w"
            ) as outfile:
                json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)
        if config["OUTPUT"].getboolean("debug_messages"):
            print(
                str(len(selected_papers))
                + " papers after title and abs filtering"
            )
            print("Total cost: $" + str(all_cost))
