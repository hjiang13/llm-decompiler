import os
import json

# Directories for source code and LLVM IR files
source_dir = "../../../../data2/jzhu/LLMs-in-IR/Human_xBenchmarks_164"  # Contains folders like CPP_0, CPP_1, ..., CPP_164 with CPP_0.cpp, etc.
ir_dir = "../../../../data2/jzhu/LLMs-in-IR/processed_data/Human_x_164_O3"  # Contains folders like CPP_0, CPP_1, ..., CPP_164 with CPP_0.bc, etc.

# Output JSONL file
output_file = "../benchmarks/dataset.jsonl"

# File naming conventions
source_ext = ".cpp"
ir_ext = ".bc"
folder_prefix = "CPP_"

# Total number of file pairs
total_files = 164

# Instruction sentence to prepend to both input and target fields
instruction1 = "Convert this to C++:\n"
instruction2 = "This is the targeted source code:\n"

with open(output_file, "w", encoding="utf-8") as out_file:
    for i in range(total_files):
        folder_name = f"{folder_prefix}{i}"
        
        # Construct full paths for the source and IR files
        source_file = os.path.join(source_dir, folder_name, f"{folder_name}{source_ext}")
        ir_file = os.path.join(ir_dir, folder_name, f"{folder_name}{ir_ext}")
        
        # Read the C++ source code
        with open(source_file, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        # Read the LLVM IR code
        with open(ir_file, "r", encoding="utf-8") as f:
            ir_code = f.read()
        
        # Create an entry with the instruction added to both fields
        entry = {
            "input": instruction1 + ir_code,
            "target": instruction2 + source_code
        }
        
        # Write the JSON object as a new line in the JSONL file
        out_file.write(json.dumps(entry) + "\n")

print(f"Dataset successfully written to {output_file}")
