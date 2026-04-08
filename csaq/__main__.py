import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import CSAQConfig
from .core import quantize
from .utils import build_calibration_data, generate_csaq_report, export_csaq_model

def main():
    parser = argparse.ArgumentParser(description="CSAQ: Causal Salience-Aware Quantization CLI")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path to quantize")
    parser.add_argument("--wbits", type=float, default=4.0, help="Target average bit-width")
    parser.add_argument("--options", type=str, default="1,2,4,8,16", help="Comma-separated allowed bit options")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the safetensors model")
    
    args = parser.parse_args()
    
    bit_options = [int(b.strip()) for b in args.options.split(",")]
    print(f"Loading {args.model_path}...")
    
    # Normally we load the model here
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Failed to load model from {args.model_path}: {e}")
        print("Continuing with dummy mode for dry-run functionality.")
        sys.exit(1)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Building calibration data (N=32)...")
    calib_data = build_calibration_data(tokenizer, n=32, seq_len=128)
    
    config = CSAQConfig(target_bits=args.wbits, bit_options=bit_options)
    model, info = quantize(model, calib_data, config=config, verbose=True)
    
    report_path = f"{args.save_path}/CSAQ_Report.json"
    generate_csaq_report(info, save_path=report_path)
    export_csaq_model(model, config, info["budget"], args.save_path)
    
if __name__ == "__main__":
    main()
