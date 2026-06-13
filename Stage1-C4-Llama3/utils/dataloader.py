import torch
from torch.utils.data import IterableDataset, get_worker_info
import glob
import os
import gzip

import datasets
import datasets.distributed

import itertools
from loguru import logger
from transformers import AutoTokenizer


def setup_dataset(args, global_rank, world_size):
    # Load from local files directly to avoid network issues
    logger.info(f"Data path: {args.data_path}")
    
    if os.path.isdir(args.data_path):
        # Local directory: load JSON.gz files directly
        # Only load train files, exclude validation files
        json_gz_pattern = os.path.join(args.data_path, "c4-train.*.json.gz")
        json_pattern = os.path.join(args.data_path, "c4-train.*.json")

        data_files_gz = sorted(glob.glob(json_gz_pattern))
        data_files_json = sorted(glob.glob(json_pattern))

        # Fallback to all json.gz if no c4-train files found
        if not data_files_gz and not data_files_json:
            json_gz_pattern = os.path.join(args.data_path, "*.json.gz")
            json_pattern = os.path.join(args.data_path, "*.json")
            data_files_gz = sorted(glob.glob(json_gz_pattern))
            data_files_json = sorted(glob.glob(json_pattern))
        
        if data_files_gz:
            logger.info(f"Loading {len(data_files_gz)} JSON.gz files from local directory: {args.data_path}")
            # Workaround for datasets library fsspec protocol tuple issue
            # Use data_dir with pattern instead of explicit file list
            try:
                data = datasets.load_dataset(
                    "json", 
                    data_dir=args.data_path,
                    data_files="*.json.gz",
                    split="train", 
                    streaming=True
                )
                logger.info("Successfully loaded data from local JSON.gz files")
            except Exception as e:
                logger.warning(f"Failed to load with data_dir pattern: {e}")
                # Fallback: try with explicit file list using absolute paths as strings
                try:
                    data_files_gz_abs = [str(os.path.abspath(f)) for f in data_files_gz]
                    data = datasets.load_dataset(
                        "json", 
                        data_files={"train": data_files_gz_abs}, 
                        split="train", 
                        streaming=True
                    )
                    logger.info("Successfully loaded data with absolute paths")
                except Exception as e2:
                    logger.warning(f"Failed to load with absolute paths: {e2}")
                    # Final fallback: try with relative paths
                    try:
                        data = datasets.load_dataset(
                            "json", 
                            data_files={"train": data_files_gz}, 
                            split="train", 
                            streaming=True
                        )
                        logger.info("Successfully loaded data from local JSON.gz files (relative paths)")
                    except Exception as e3:
                        logger.error(f"Failed to load local files: {e3}")
                        raise
        elif data_files_json:
            logger.info(f"Loading {len(data_files_json)} JSON files from local directory: {args.data_path}")
            # Workaround for datasets library fsspec protocol tuple issue
            # Use data_dir with pattern instead of explicit file list
            try:
                data = datasets.load_dataset(
                    "json", 
                    data_dir=args.data_path,
                    data_files="*.json",
                    split="train", 
                    streaming=True
                )
                logger.info("Successfully loaded data from local JSON files")
            except Exception as e:
                logger.warning(f"Failed to load with data_dir pattern: {e}")
                # Fallback: try with explicit file list using absolute paths as strings
                try:
                    data_files_json_abs = [str(os.path.abspath(f)) for f in data_files_json]
                    data = datasets.load_dataset(
                        "json", 
                        data_files={"train": data_files_json_abs}, 
                        split="train", 
                        streaming=True
                    )
                    logger.info("Successfully loaded data with absolute paths")
                except Exception as e2:
                    logger.warning(f"Failed to load with absolute paths: {e2}")
                    # Final fallback: try with relative paths
                    try:
                        data = datasets.load_dataset(
                            "json", 
                            data_files={"train": data_files_json}, 
                            split="train", 
                            streaming=True
                        )
                        logger.info("Successfully loaded data from local JSON files (relative paths)")
                    except Exception as e3:
                        logger.error(f"Failed to load local JSON files: {e3}")
                        raise
        else:
            raise ValueError(f"No JSON or JSON.gz files found in {args.data_path}")
    else:
        # HuggingFace dataset name: use original method
        data = datasets.load_dataset(args.data_path, args.data_name, split="train", streaming=True)

    seed_for_shuffle = 42 

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # T5 tokenizer trained on C4; download from HuggingFace (set HF_ENDPOINT for mirror)
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base",
        model_max_length=args.max_length,
    )
    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    return dataloader, tokenizer


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.original_data = data  # Keep reference to original data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1

        # Create a fresh shuffle for each epoch with different seed
        epoch_seed = 41 + self.epoch  # Different seed for each epoch
        shuffled_data = self.original_data.shuffle(seed=epoch_seed)

        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(shuffled_data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(shuffled_data, worker_id, None, num_workers)

        batch = []
        skip_count = 0
        max_skip = 1000  # Maximum number of consecutive skips before giving up
        
        while True:
            try:
                example = next(iter_data)
                skip_count = 0  # Reset skip count on successful read
            except StopIteration:
                # End of dataset
                break
            except (gzip.BadGzipFile, OSError, IOError) as e:
                # Skip corrupted gzip files
                skip_count += 1
                if skip_count >= max_skip:
                    logger.error(f"Too many consecutive corrupted files ({max_skip}), stopping iteration")
                    break
                if skip_count % 100 == 0:
                    logger.warning(f"Skipped {skip_count} corrupted files")
                continue
            except Exception as e:
                # Log unexpected errors but continue
                logger.warning(f"Unexpected error reading example: {e}, skipping...")
                skip_count += 1
                if skip_count >= max_skip:
                    logger.error(f"Too many consecutive errors ({max_skip}), stopping iteration")
                    break
                continue
            
            try:
                # Skip examples without text field or with empty text
                if "text" not in example or not example["text"]:
                    continue
                    
                tokenized_example = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch.append(tokenized_example)

                if len(batch) == self.batch_size:
                    yield self._format_batch(batch)
                    batch = []
            except (KeyError, ValueError, TypeError) as e:
                # Skip invalid examples
                continue
            except Exception as e:
                # Log unexpected errors but continue
                logger.warning(f"Unexpected error processing example: {e}, skipping...")
                continue

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
