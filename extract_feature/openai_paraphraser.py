import json

from tqdm import tqdm
import time

from openai import OpenAI


def paraphrase_string(input_string):
    openai = OpenAI()

    # Specify the prompt for paraphrasing
    prompt = f"Paraphrase the following sentence:\n'{input_string}'\nParaphrased sentence:"

    # Use the OpenAI Completions API to generate a paraphrased version
    # response = openai.Completion.create(
    #     engine="gpt-3.5-turbo-instruct",  # You can try other engines as well
    #     prompt=prompt,
    #     temperature=0.7,
    #     max_tokens=100,
    #     n=1,
    #     stop=None,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,
        max_tokens=32,
        n=1,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the paraphrased sentence from the API response
    paraphrased_sentence = response.choices[0].text.strip()

    print(f"\n{input_string=} <-> {paraphrased_sentence=}")

    return paraphrased_sentence


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def count_words(input_string):
    # Split the string into words using spaces as separators
    words = input_string.split()

    # Count the number of words
    word_count = len(words)

    return word_count


def get_paraphrased_strings():
    input_file = "../data/highlight_train_release_paraphrased.jsonl"
    json_data = load_jsonl(input_file)
    data_dict = {}

    print("Loading data...")
    for data in tqdm(json_data):
        data_dict[f"{data['qid']}_{data['aug_id']}"] = data

    print("Processing data...")
    for data in tqdm(json_data):
        if count_words(data['query']) > 32:
            print(f"found problematic query: {data['qid']=} {data['aug_id']=} -> {data['query']}")
            if data['aug_id'] == 1:
                data['query'] = paraphrase_string(data_dict[f"{data['qid']}_0"]['query'])
                time.sleep(0.5)

    save_jsonl(json_data, input_file+"1")


def paraphrase_strings():
    input_file = "../data/highlight_train_release.jsonl"
    output_file = "../data/highlight_train_release_paraphrased_openai.jsonl"
    json_data = load_jsonl(input_file)
    augmented_data = []

    i = 0;
    print("Loading data...")
    for data in tqdm(json_data):
        new_dict = dict.fromkeys(
            ['qid', 'aug_id', 'query', 'duration', 'vid', 'relevant_clip_ids', 'saliency_scores', 'relevant_windows'])
        new_dict["qid"] = data["qid"]
        new_dict["aug_id"] = 0
        new_dict["query"] = data["query"]
        new_dict["duration"] = data["duration"]
        new_dict["vid"] = data["vid"]
        new_dict["relevant_clip_ids"] = data["relevant_clip_ids"]
        new_dict["saliency_scores"] = data["saliency_scores"]
        new_dict["relevant_windows"] = data["relevant_windows"]
        augmented_data.append(new_dict)

        new_dict = new_dict.copy()
        new_dict["aug_id"] = 1
        new_dict['query'] = paraphrase_string(new_dict['query'])
        augmented_data.append(new_dict)
        time.sleep(0.25)

    save_jsonl(augmented_data, output_file)


if __name__ == "__main__":
    # get_paraphrased_strings()
    paraphrase_strings()
