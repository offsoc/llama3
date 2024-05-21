# This supports merging as many adapters as you want.

# python merge_adapters.py --base_model_name_or_path <base_model> --peft_model_paths <adapter1> <adapter2> <adapter3> --output_dir <merged_model>

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_paths", type=str, nargs='+', help="List of paths to PEFT models")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    if args.device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": args.device}}

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        **device_arg
    )

    model = base_model

    for peft_model_path in args.peft_model_paths:
        print(f"Loading PEFT: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path, **device_arg)
        print(f"Running merge_and_unload for {peft_model_path}")
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
    else:
        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")

    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
