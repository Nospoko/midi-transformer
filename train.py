from datasets import load_dataset


#  prototype - testing dataset building
def main():
    dataset = load_dataset("./OneTimeTokDataset", name="debugging", trust_remote_code=True, num_proc=8)
    record = dataset["train"][0]
    print(len(record["note_tokens"]))
    print(record["source"])
    print(dataset["train"].info.tokenizer_name)


if __name__ == "__main__":
    main()
