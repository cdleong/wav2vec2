import json
import re
from pathlib import Path

from tqdm.auto import tqdm
import torch
import torchaudio
import pandas as pd
import pyarrow.parquet as pq

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from argument_classes import ModelArguments, DataTrainingArguments

import ftfy
import unicodedata
import jiwer

import librosa

import warnings
warnings.filterwarnings("ignore")

import random


args_file = './args.json'
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(args_file)

# increasing number of threads for torchaudio resample
print(f'Using {data_args.preprocessing_num_workers} threads')
torch.set_num_threads(data_args.preprocessing_num_workers)

# Get the datasets:
train_dataset = datasets.load_dataset(
    "common_voice", data_args.dataset_config_name, split=data_args.train_split_name
)
eval_dataset = datasets.load_dataset("common_voice", data_args.dataset_config_name, split="test")

print(train_dataset.features)

print("original train_dataset length")
length_original = len(train_dataset)
print(len(train_dataset))


eval_original_length = len(eval_dataset)

print('filter items with small numbers of downvotes from dataset objects')
train_dataset = train_dataset.filter(lambda example: example["down_votes"] == 0)
eval_dataset = eval_dataset.filter(lambda example: example["down_votes"] == 0)

print("now train_dataset is this big:")
length_after_downvotes = len(train_dataset)
print(len(train_dataset))

eval_length_after_downvotes = len(eval_dataset)






# # Filter by max length
# # copied from https://github.com/elgeish/transformers/blob/820de495fcfc05da447f926189473ffc2cf0a5cb/examples/research_projects/wav2vec2/run_asr.py#L372
# print("filtering by max duration")
# def prepare_to_filter_by_max_length(example):
#     with warnings.catch_warnings():  # prevents warning with PySoundFile
#         warnings.simplefilter("ignore")
#         example["speech"], example["sampling_rate"] = librosa.load(example['path'], sr=16_000)
#         example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]    


# train_dataset = train_dataset.map(prepare_to_filter_by_max_length, num_proc=data_args.preprocessing_num_workers)    
# eval_dataset = eval_dataset.map(prepare_to_filter_by_max_length, num_proc=data_args.preprocessing_num_workers)    

# def filter_by_max_duration(example):
#     return example["duration_in_seconds"] <= 15


# train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"], num_proc=data_args.preprocessing_num_workers)   
# eval_dataset = eval_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"], num_proc=data_args.preprocessing_num_workers)   

# print("now train_dataset is this big:")
# length_after_max_duration = len(train_dataset)
# print(len(train_dataset))






    
# Create and save tokenizer
#chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\(\)\/\®\_\©\√\«\[\]\{\}\™\‽\…\‟\ˮ\„\″\¸\»\·\•\˝\˜˜\ʺ\|\—\¬\~\¨\ß\#\€\*\+\<\>\=\¤\$\ª\£\°\‚\|]'
arabic_characters = r'[\ە\ش\ب]'
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def remove_special_characters(batch):

    
    batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower() + " "

#     debug = False
#     if 'ł' in batch["sentence"]:
#         print(batch)
#         print(batch["sentence"])
#         print(batch["text"])
#         print("BEFORE")
#         debug = True 
        
#     # ҫ
#     if 'ҫ' in batch["sentence"]:
#         print(batch)
#         print(batch["sentence"])
#         print(batch["text"])
#         print("BEFORE")
#         debug = True 
            
    
    
    
    # remove the three arabic characters: 'ب': 42, 'ش': 6, 'ە': 21,
    # not even pronounced in common_voice_rw_23520407.mp3	بەش na none	2	0	twenties			rw
    batch["text"] = re.sub(arabic_characters, '', batch["text"])

    # replace special version of characters

    # в found in common_voice_rw_23200846.mp3	ni umweкre mu baвri
    batch["text"] = re.sub('в', 'b', batch["text"])


    # ﬁ common_voice_rw_23158647.mp3	hagati y’abaﬁte imyanya runaka
    batch["text"] = re.sub('ﬁ', 'fi', batch["text"])

    # 'к': 1, common_voice_rw_23200846.mp3	ni umweкre mu baвri
    batch["text"] = re.sub('к', 'k', batch["text"])

    # о
    batch["text"] = re.sub('о', 'o', batch["text"])

    # р -> p
    batch["text"] = re.sub('р', 'p', batch["text"])

    # с -> c
    batch["text"] = re.sub('с', 'c', batch["text"])

    # у -> y
    batch["text"] = re.sub('у', 'y', batch["text"])

    # і -> i
    batch["text"] = re.sub('і', 'i', batch["text"])


    # ø found in common_voice_rw_22558611.mp3	Køge North Station naryo n’irindi, just ronounced as O
    batch["text"] = re.sub('ø', 'o', batch["text"])

    # ҫ found in common_voice_rw_22577788.mp3	Ati “Albert yari maneko nari muzi akiga muri Rambura Garҫons
    batch["text"] = re.sub('ҫ', 'c', batch["text"])

    # 'м': 36, found in  common_voice_rw_22629158.mp3	Lukа Моdrіс (yaguzwe muri Tottenham Hotspur)
    batch["text"] = re.sub('м', 'm', batch["text"])

    # ф found in common_voice_rw_23044503.mp3	akagira isura idasebya isфoko, just pronounced as "o"
    batch["text"] = re.sub('ф', 'o', batch["text"])


    #  '¯': 51, not pronounced in common_voice_rw_23013758.mp3	® akadahera ni urwimo n’urugaryi ¯
    batch["text"] = re.sub('¯', '', batch["text"])

    #  '–': 21, found in many places, not pronounced.
    batch["text"] = re.sub('–', '', batch["text"])

    #  '―': 35, not pronounced in common_voice_rw_23441475.mp3	cyangwa se mudaherukanye umusuhuza ukoresheje ― Muraho
    batch["text"] = re.sub('―', '', batch["text"])

    #  '−': 36}
    batch["text"] = re.sub('−', '', batch["text"])

    # ₋ only shows up once, drop
    batch["text"] = re.sub('₋', '', batch["text"])

    # ‐ can be dropped too
    batch["text"] = re.sub('‐', '', batch["text"])


    # normalize apostrophes
    #  '`': 31,
    batch["text"] = re.sub('`', '\'', batch["text"])
    #  '´': 21,
    batch["text"] = re.sub('´', '\'', batch["text"])
    #  'ʻ': 10,
    batch["text"] = re.sub('ʻ', '\'', batch["text"])
    #  'ʽ': 58,
    batch["text"] = re.sub('ʽ', '\'', batch["text"])
    #  '΄': 61,
    batch["text"] = re.sub('΄', '\'', batch["text"])
    
    # ʼ
    batch["text"] = re.sub('ʼ', '\'', batch["text"])
    # ’
    batch["text"] = re.sub('’', '\'', batch["text"])

    # double-singlequotes go away entirely
    batch["text"] = re.sub("''", '', batch["text"])

#     # |
#     batch["text"] = re.sub("|", ' ', batch["text"])
    

    # ¼ is pronounced. Sounds like "chimecha" to me. ½ is as well. ¾ seems to be also.

    # ð only shows up once, and doesn't seem necessary. Drop.
    batch["text"] = re.sub("ð", '', batch["text"])

    # ł shows up in a list of Polish place names. replace with l
    batch["text"] = re.sub("ł", 'l', batch["text"])

    # - not pronounced, mostly. e.g. common_voice_rw_21012753.mp3	-N’uko ibibazo bya politiki n’umutekano
    batch["text"] = re.sub('-', '', batch["text"])

    # æ found in common_voice_rw_23060030.mp3	"muri æon y'ibirimo!" and common_voice_rw_23330049.mp3	"pæan y'urukundo rumwe reverberant.
    batch["text"] = re.sub('æ', 'ae', batch["text"])

    # œ shows up in "fœtus" and what looks like corrupted text.
    batch["text"] = re.sub('œ', 'oe', batch["text"])

    # @ is pronounced, but there's only 15 examples. Hmmmm... let's replace
    batch["text"] = re.sub('@', ' at ', batch["text"])

    # \\ doesn't seem to be pronounced.
    batch["text"] = re.sub("\\\\", ' at ', batch["text"])

    # from https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    batch["text"] = strip_accents(batch["text"])

    # just for good measure
    batch["text"] = jiwer.RemovePunctuation()(batch["text"])

    
    
    batch["sentence"] = batch["text"]
    
#     if debug:
#         print(batch)
#         print(batch["sentence"])
#         print(batch["text"])
#         print("AFTER")
#         exit()
    
    return batch



print("***********************")
print("REMOVING SPECIAL CHARACTERS")
train_dataset = train_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True, num_proc=data_args.preprocessing_num_workers)
eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True, num_proc=data_args.preprocessing_num_workers)

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
vocab_test = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)



vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)



if data_args.max_train_samples is not None:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))

if data_args.max_val_samples is not None:
    eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

# Load pretrained tokenizer & create processor
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)




feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)







# =======================================================
# The following part is modified to:
#   - load, resample, process and save audio files into raw tensors
#   - process labels as before
#   - save datasets containing the paths and labels to disk, as arrow table in parquet format
#   - save processor to disk, for reuse in training script (I got a hint that it is not deterministic)

# load and resample audio, save as raw tensors
resampler = torchaudio.transforms.Resample(48_000, 16_000)
resampled_data_dir = Path('/workspace/.cache/resampled')
resampled_data_dir.mkdir(exist_ok=True)


max_length_seconds = 15
bad_paths = []
too_long_paths = []

def exceeds_max_length(speech_array, sampling_rate, max_duration_seconds=15):
    exceeds_max_length = False
    duration_in_seconds = len(speech_array)/sampling_rate
    if duration_in_seconds > max_duration_seconds:
        exceeds_max_length = True
        
    return exceeds_max_length

    

def load_resample_save(f):
    f = Path(f)
    new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
    if not new_path.exists():
        speech_array, sampling_rate = torchaudio.load(f)
        if exceeds_max_length(speech_array, sampling_rate):
            too_long_paths.append(str(new_path))
            return ""
        
        
        # remove files that are too long, so as to not get out of memory errors. 
        # inspired by https://github.com/serapio/transformers/blob/e17bf2a0e99935e58c50adf6ce70ce68926cc1b3/examples/research_projects/wav2vec2/run_common_voice.py#L511
        # train_dataset = train_dataset.filter(lambda batch: len(batch["speech"]) < 150000) # this is how they did it in example above.  
        

        # https://github.com/elgeish/transformers/blob/820de495fcfc05da447f926189473ffc2cf0a5cb/examples/research_projects/wav2vec2/run_asr.py#L375 has another version
#         example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        
        
        speech_array_resampled = resampler(speech_array)
        input_values = processor(speech_array_resampled, sampling_rate=16_000).input_values
        input_values = torch.from_numpy(input_values).float().flatten()
        torch.save(input_values, new_path)

#     # TODO: move this check to the if statement above. 
#     speech_array, sampling_rate = torchaudio.load(f)
#     if exceeds_max_length(speech_array, sampling_rate):
#         new_path.unlink()
#         too_long_paths.append(str(new_path))
#         return ""
    

#     speech_array, sampling_rate = torchaudio.load(f)
#     # remove files that are too long, so as to not get out of memory errors. 
#     # inspired by https://github.com/serapio/transformers/blob/e17bf2a0e99935e58c50adf6ce70ce68926cc1b3/examples/research_projects/wav2vec2/run_common_voice.py#L511
#     # train_dataset = train_dataset.filter(lambda batch: len(batch["speech"]) < 150000) # this is how they did it in example above.  
#     if len(speech_array) > 150000:
# #         print(f"{f} was too long!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         new_path.unlink()  # remove the file
#         bad_paths.append(str(new_path))
# #         print(f"{f} was too long!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# #         exit()
#         return ""    
    
    # remove empty files. 
    if new_path.stat().st_size > 0: # TODO: move the file size check to the end?
        return str(new_path)
    else: 
#         print(f"{f} was empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        bad_paths.append(str(new_path))
#         return str(new_path)
        return ""
        

    
        
print('load resample save')
new_train_paths = [load_resample_save(f) for f in tqdm(train_dataset['path'], miniters=100, desc='train')]
new_eval_paths = [load_resample_save(f) for f in tqdm(eval_dataset['path'], miniters=100, desc='eval')]

print("now train_dataset is this big:")
print(len(train_dataset))
length_after_resample = len(train_dataset)

# print('remove_empty_files from list')
# new_train_paths = [i for i in tqdm(new_train_paths, miniters=100, desc='train') if i]
# new_eval_paths = [i for i in tqdm(new_eval_paths, miniters=100, desc='eval') if i]




# update paths and sampling rate
train_dataset = train_dataset.map(
    lambda x: {'path': new_train_paths, 'sampling_rate':[16_000] * len(train_dataset), 'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    lambda x: {'path': new_eval_paths, 'sampling_rate':[16_000] * len(eval_dataset), 'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)

# tokenize targets
def tokenize_targets(batch):
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch




print('preparing dataset: train')
train_dataset = train_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in train_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)
print('preparing dataset: eval')
eval_dataset = eval_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in eval_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)



# https://huggingface.co/docs/datasets/v0.3.0/processing.html#filtering-rows-select-and-filter
def keep_sample(example):
    f = example['path']
    
    if f: # I set it to "" earlier. TODO: do the actual file size checking here, for aesthetic reasons. 
        return True
    else: 
        return False


print('filter bad paths and too-long paths from dataset objects')
print(f"bad paths: {bad_paths}")
train_dataset = train_dataset.filter(keep_sample)
eval_dataset = eval_dataset.filter(keep_sample)

print("now train_dataset is this big:")
print(len(train_dataset))
length_after_bad_paths_and_too_long = len(train_dataset)
eval_length_after_bad_paths_and_too_long = len(eval_dataset)


# # save for disk, ready for training
pq.write_table(train_dataset.data, f'./{data_args.dataset_config_name}.train.parquet')
pq.write_table(eval_dataset.data, f'./{data_args.dataset_config_name}.eval.parquet')

# save processor for training
processor.save_pretrained(training_args.output_dir)

print(f"length_original {length_original}")
print(f"length_after_downvotes {length_after_downvotes}")
# print(f"length_after_max_duration {length_after_max_duration}")
print(f"length_after_resample: {length_after_resample}")
print(f"length_after_bad_paths_and_too_long: {length_after_bad_paths_and_too_long}")
print(f"len(too_long_paths): {len(too_long_paths)}")
if len(too_long_paths) > 0:
    print(f"random too-long file: {random.choice(too_long_paths)}")
    
print(f"len(bad_paths): {len(bad_paths)}")    
if len(bad_paths) > 0:
    print(f"random empty file: {random.choice(bad_paths)}")
    

print(f"eval_original_length: {eval_original_length} ")
print(f"eval_length_after_downvotes: {eval_length_after_downvotes} ")
print(f"eval_length_after_bad_paths_and_too_long: {eval_length_after_bad_paths_and_too_long} ")