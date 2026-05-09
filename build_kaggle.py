import os

def main():
    with open("examples/konkani_sample.txt", "r", encoding="utf-8") as f:
        konkani_text = f.read()
        
    with open("benchmarks/research_validation.py", "r", encoding="utf-8") as f:
        research_code = f.read()
        
    research_code_escaped = research_code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    
    kaggle_code = '''# ==============================================================================
# CSAQ KAGGLE RESEARCH NOTEBOOK SCRIPT
# Copy and paste this entire block into a single Kaggle Notebook cell.
# ==============================================================================

import os
import subprocess
import sys

# 1. Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "csaq-quant==0.5.2", "datasets", "scipy", "matplotlib"], check=True)

# 2. Write the Konkani data to disk
os.makedirs("data", exist_ok=True)
print("Writing Konkani corpus...")
konkani_text = """{konkani_text}"""
with open("data/konkani_sample.txt", "w", encoding="utf-8") as f:
    f.write(konkani_text)

# 3. Download and prepare English dataset (WikiText-2)
print("Downloading WikiText-2...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split, filename in [("train", "wikitext2_train.txt"), ("test", "wikitext2_test.txt")]:
        lines = [t for t in ds[split]["text"] if len(t.strip()) > 20]
        with open(f"data/{filename}", "w", encoding="utf-8") as f:
            f.writelines(line + "\\n" for line in lines)
except Exception as e:
    print(f"Failed to load WikiText: {e}")

# 4. Create the research validation script locally
print("Writing validation logic...")
research_script = """{research_script}"""
with open("research_validation.py", "w", encoding="utf-8") as f:
    f.write(research_script)

# 5. Run the experiment
print("\\n" + "="*80)
print("STARTING CSAQ EXPERIMENT")
print("="*80 + "\\n")

# NOTE: Using a 0.5B model by default to fit easily in Kaggle T4x2.
cmd = [
    sys.executable, "research_validation.py",
    "--model_path", "Qwen/Qwen1.5-0.5B",
    "--calib_file_domain", "data/konkani_sample.txt",
    "--eval_file_domain", "data/konkani_sample.txt",
    "--calib_file_english", "data/wikitext2_train.txt",
    "--eval_file_english", "data/wikitext2_test.txt",
    "--domain_name", "konkani",
    "--output_dir", "./csaq_research_output",
    "--target_bits", "4.0",
    "--device", "auto"
]

subprocess.run(cmd, check=True)
print("\\nExperiment completed! Check the ./csaq_research_output directory for CSVs and figures.")
'''.replace("{konkani_text}", konkani_text).replace("{research_script}", research_code_escaped)
    with open("kaggle_notebook_cell.py", "w", encoding="utf-8") as f:
        f.write(kaggle_code)
        
if __name__ == "__main__":
    main()
