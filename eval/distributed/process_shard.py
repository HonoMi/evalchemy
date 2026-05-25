import argparse
import logging
import os
import re
import tempfile

from datasets import load_dataset
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HARMONY_FINAL_PREFIX_RE = re.compile(r"^\s*(?:assistant\s+)?final\b[:\s]*", re.IGNORECASE)


def _strip_harmony_final_prefix(text: str) -> str:
    return _HARMONY_FINAL_PREFIX_RE.sub("", text, count=1)


def _supports_harmony_postprocess(tokenizer) -> bool:
    return hasattr(tokenizer, "parse_response") or hasattr(tokenizer, "parse_harmony_message")


def _extract_harmony_content(tokenizer, token_ids: list[int], fallback_text: str) -> str:
    if not _supports_harmony_postprocess(tokenizer):
        return fallback_text

    decoded_text = fallback_text
    if token_ids:
        try:
            decoded_text = tokenizer.decode(token_ids)
        except Exception as exc:
            logger.debug("Failed to decode Harmony token ids: %r", exc)

    if hasattr(tokenizer, "parse_response"):
        try:
            parsed = tokenizer.parse_response(decoded_text)
            content = parsed.get("content") if isinstance(parsed, dict) else None
            if isinstance(content, str) and content.strip():
                return content
        except Exception as exc:
            logger.debug("Failed to parse Harmony response: %r", exc)

    if token_ids and hasattr(tokenizer, "parse_harmony_message"):
        try:
            response_prefill = tokenizer.encode("<|start|>assistant")
            parsed_messages = tokenizer.parse_harmony_message(response_prefill + token_ids)
            last_content = None
            for message in parsed_messages:
                content = getattr(message, "content", None)
                if content is not None and getattr(content, "token_ids", None):
                    text = tokenizer.decode(content.token_ids)
                    if text.strip():
                        last_content = text
            if last_content is not None:
                return last_content
        except Exception as exc:
            logger.debug("Failed to parse Harmony token stream: %r", exc)

    return _strip_harmony_final_prefix(decoded_text)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=30, max=600),
    reraise=True,
)
def upload_shard(dataset, output_dataset, shard_num, num_shards):
    """Push dataset shard to Hugging Face Hub with automatic retries.

    Args:
        dataset: The dataset object to upload
        output_dataset: The Hugging Face Hub repository ID to upload to
        shard_num: The index of the current shard
        num_shards: The total number of shards
    """
    api = HfApi()
    # Check if repo exists before creating
    try:
        api.repo_info(repo_id=output_dataset, repo_type="dataset")
        print(f"Repository {output_dataset} already exists")
    except Exception:
        print(f"Creating new repository {output_dataset}")
        api.create_repo(repo_id=output_dataset, repo_type="dataset")

    try:
        # Create temporary file and save the dataset
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            dataset.to_parquet(tmp.name)
            # Format the filename for the shard
            shard_filename = f"train-{shard_num:05d}-of-{num_shards:05d}.parquet"
            # Upload the file
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=shard_filename,
                repo_id=output_dataset,
                repo_type="dataset",
                commit_message=f"Adding shard {shard_num}",
            )

        print(f"Successfully pushed shard {shard_num} to {output_dataset} as {shard_filename}")
    except Exception as e:
        print(f"Push failed for shard {shard_num}, will retry: {str(e)}")
        raise


def process_shard(
    input_dataset: str,
    rank: int,
    global_size: int,
    model_name: str,
    tp: int,
    output_dataset: str,
    upload: bool = False,
) -> None:
    """Process a single shard of the dataset using VLLM for inference.

    Args:
        input_dataset: The Hugging Face Hub repository ID containing the inputs
        rank: The shard index (0 to global_size-1)
        global_size: Total number of shards
        model_name: The name or path of the model for VLLM
        tp: Tensor parallelism size for VLLM
        output_dataset: Repository ID or local directory to save processed data
        upload: If True, upload to Hugging Face Hub; if False, save locally
    """
    # Load the dataset from Hugging Face Hub
    logger.info(f"Loading dataset from {input_dataset}")
    ds = load_dataset(input_dataset, split="train")
    ds = ds.shuffle(seed=42)  # shuffle first for fair sharding

    # Shard the dataset
    logger.info(f"Sharding dataset: {global_size} shards, processing shard {rank}")
    ds = ds.shard(num_shards=global_size, index=rank)

    # Initialize VLLM
    logger.info(f"Using model: {model_name}")
    logger.info("Initializing VLLM")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp,
    )

    # Initialize tokenizer for chat templates
    logger.info("Initializing tokenizer for chat template application")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Preprocess the examples
    sampling_params = []
    prompts = []
    logger.info(f"Applying chat template and getting sampling params for {len(ds)} examples")
    for example in ds:
        # Create SamplingParams from gen_kwargs
        gen_kwargs = example.get("gen_kwargs", {})
        params = SamplingParams(
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.95),
            max_tokens=gen_kwargs.get("max_new_tokens", 4096),
            stop=gen_kwargs.get("stop", None),
            seed=gen_kwargs.get("seed", None),
        )
        sampling_params.append(params)

        # Apply chat template
        messages = example["context"]
        assert all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages)
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    # Generate outputs with vLLM
    logger.info(f"Generating outputs for {len(prompts)} examples")
    outputs = llm.generate(prompts, sampling_params)

    # Process outputs and store results
    outputs_text = [
        _extract_harmony_content(
            tokenizer,
            list(getattr(output.outputs[0], "token_ids", []) or []),
            output.outputs[0].text,
        )
        for output in outputs
    ]
    ds = ds.add_column("model_outputs", outputs_text)
    logger.info(f"Shard successfully processed and loaded into dataset: {len(ds)} examples")

    # Upload to HF Hub or save locally
    if upload:
        try:
            upload_shard(ds, output_dataset, rank, global_size)
            logger.info(f"Shard {rank} pushed to hub as {output_dataset}")
            logger.info(f"View the dataset at https://huggingface.co/datasets/{output_dataset}")
        except Exception as e:
            logger.error(f"Failed to push shard {rank} after all retries: {str(e)}")
    else:
        shard_filename = f"train-{rank:05d}-of-{global_size:05d}.parquet"
        os.makedirs(output_dataset, exist_ok=True)
        local_path = os.path.join(output_dataset, shard_filename)
        ds.to_parquet(local_path)
        logger.info(f"Saved shard {rank} locally to {local_path}")


def main():
    """Parse command line arguments and run the sharded inference process."""
    parser = argparse.ArgumentParser(description="Run VLLM inference on a sharded dataset")
    parser.add_argument("--global_size", type=int, required=True, help="Total number of shards")
    parser.add_argument("--rank", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--input_dataset", type=str, required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--output_dataset", type=str, required=True, help="HF repo id or local dir path")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model for VLLM")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size for VLLM")
    parser.add_argument("--upload", action="store_true", help="Upload results to Hugging Face Hub")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without internet access")

    args = parser.parse_args()
    parsed_log = "\n".join([x.upper() + ": " + str(v) for x, v in vars(args).items()])
    logger.info(f"Parsed arguments:\n{parsed_log}")

    # Set offline mode if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
        logger.info(f"Running in offline mode, will use $HF_HUB_CACHE: {HF_HUB_CACHE} as cache directory")
        if args.upload:
            raise ValueError("Cannot upload in offline mode")

    # Validate arguments
    if args.rank < 0 or args.rank >= args.global_size:
        raise ValueError(f"Rank ({args.rank}) must be between 0 and global_size-1 ({args.global_size-1})")

    # Process the shard
    process_shard(
        args.input_dataset,
        args.rank,
        args.global_size,
        args.model_name,
        args.tp,
        args.output_dataset,
        args.upload,
    )


if __name__ == "__main__":
    main()
