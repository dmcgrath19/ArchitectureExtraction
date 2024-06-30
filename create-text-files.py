from datasets import load_dataset

def parse_pilecorpus(path,start_seed=42):
    """
    This is a way for parsing the Pile corpus.
    """
    print("Streaming the main pile corpus")
    
    all_texts = ""
    dataset = load_dataset(path, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(100000)

    for text in dataset_head:
        all_texts+= text['text']

    return all_texts

def parse_splitted(path, subset='default', max_examples=1000000, start_seed=42):
    """
    This is for parsing thePileSplitted dataset.
    """
    print("Streaming the splitted pile")
    
    all_texts = ""
    examples_processed = 0

    print(f"Subset: {subset}")
    print(f"Path: {path}")

    # Load the dataset subset with streaming enabled
    dataset = load_dataset(path, subset,split="train", streaming=True)

    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(100000)

    for text in dataset_head:
        all_texts+= text['text']

    print("completed parsing")

    return all_texts



def parse_wmt_splitted(path, split_set='train', start_seed=33):
    """
    This is for getting data from KaiNylund/WMT-year-splits
    unseen data for the model serving as a base for perplexity
    """

    print("Streaming the wmt splitted dataset")

    all_texts = ""
    
    # Load the dataset split with streaming enabled
    dataset = load_dataset(path, split=split_set, streaming=True)
    
    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(100000)

    for text in dataset_head:
        all_texts+= text['text']

    print("completed parsing")
    
    return all_texts

git = parse_splitted(path="ArmelR/the-pile-splitted", subset="Github")
wiki = parse_splitted(path="ArmelR/the-pile-splitted", subset="Wikipedia (en)")
dm = parse_splitted(path="ArmelR/the-pile-splitted", subset="DM Mathematics")

wmt = parse_wmt_splitted(path='KaiNylund/WMT-year-splits', split_set="2021_train")

with open('wmt.txt', 'w') as file:
    # Write the string to the file
    file.write(wmt)

with open('git.txt', 'w') as file:
    # Write the string to the file
    file.write(git)

with open('wiki.txt', 'w') as file:
    # Write the string to the file
    file.write(wiki)

with open('dm.txt', 'w') as file:
    # Write the string to the file
    file.write(dm)

