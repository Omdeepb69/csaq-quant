import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csaq import quantize, CSAQConfig, build_calibration_data

def main():
    print("Loading Model...")
    model_id = "gpt2" # Using a small model for demonstration. Replace with your target model.
    
    # Best practice is to load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    
    # 1. Build Calibration Data
    # 32 samples are usually enough to hit the Spearman Rank Early Stopping threshold
    print("Building Calibration Data...")
    calib_data = build_calibration_data(
        tokenizer=tokenizer,
        n=32, 
        seq_len=128, 
        dataset="wikitext"
    )

    # 2. Configure CSAQ Constraint Solver
    # We aim for exactly 4.0 bits/weight on average
    print("Configuring CSAQ Solver...")
    config = CSAQConfig(
        target_bits=4.0, 
        bit_options=[1, 2, 4, 8, 16],
        clique_threshold=0.85
    )

    # 3. Trigger 3-Phase Quantization Engine
    # - Phase 1: Causal Salience Profiling
    # - Phase 2: Jaccard Graph Interaction Discovery
    # - Phase 3: Fractional Budget Allocation
    print("Executing Quantization Engine...")
    quantized_model, info = quantize(
        model=model,
        calib_data=calib_data,
        config=config,
        verbose=True
    )
    
    # Print the resulting bit distribution logic
    for bits, count in info["tier_stats"].items():
        print(f"Assigned to {bits}: {count} elements")

    # You can now save it using the utils or your own mechanisms:
    # from csaq.utils import generate_csaq_report, export_csaq_model
    # generate_csaq_report(info, save_path="./CSAQ_Report.json")
    # export_csaq_model(quantized_model, config, info["budget"], "./csaq_output")

if __name__ == "__main__":
    main()
