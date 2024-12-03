import os
import sys
import logging
from fault_localizer_trainer import train_fault_localizer, predict_fault_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_code_paths():
    """Get paths to code and output directories"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
   
    # Code path is in ConDefects/Code
    code_dir = os.path.join(current_dir, "Code")
    if not os.path.exists(code_dir):
        logger.error(f"Code directory not found at {code_dir}")
        sys.exit(1)
       
    # Create output directory for model
    output_dir = os.path.join(current_dir, "fault_localizer_output")
    os.makedirs(output_dir, exist_ok=True)
   
    return code_dir, output_dir

def main():
    # Get directory paths
    code_dir, output_dir = get_code_paths()
    logger.info(f"Using code directory: {code_dir}")
    logger.info(f"Using output directory: {output_dir}")
   
    # Train the model
    model, tokenizer = train_fault_localizer(
        programs_dir=code_dir,
        output_dir=output_dir,
        model_name="microsoft/codebert-base"
    )
   
    # Example: Let's test it on a specific program
    # Find a Python program to test
    for contest in os.listdir(code_dir):
        contest_dir = os.path.join(code_dir, contest)
        python_dir = os.path.join(contest_dir, "Python")
        if os.path.exists(python_dir):
            for program in os.listdir(python_dir):
                program_dir = os.path.join(python_dir, program)
                faulty_file = os.path.join(program_dir, "faultyVersion.py")
                if os.path.exists(faulty_file):
                    logger.info(f"\nTesting model on: {faulty_file}")
                    predictions = predict_fault_location(model, tokenizer, faulty_file)
                   
                    print("\nTop 5 most suspicious lines:")
                    for line_no, line_content, probability in predictions[:5]:
                        print(f"Line {line_no} ({probability:.3f}): {line_content}")
                       
                    # Read actual fault location
                    fault_loc_file = os.path.join(program_dir, "faultLocation.txt")
                    if os.path.exists(fault_loc_file):
                        with open(fault_loc_file, 'r') as f:
                            actual_fault = int(f.read().strip())
                        print(f"\nActual fault is at line: {actual_fault}")
                       
                    break
            break

if __name__ == "__main__":
    main()
