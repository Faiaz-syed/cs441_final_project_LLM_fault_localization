import os
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_prepare_data(code_dir):
    """
    Check if data is properly structured and count available programs
    """
    if not os.path.exists(code_dir):
        logger.error(f"Code directory not found: {code_dir}")
        return False
        
    python_programs = []
    total_contests = 0
    python_contests = 0
    
    # Check each contest directory
    for contest in os.listdir(code_dir):
        contest_dir = os.path.join(code_dir, contest)
        if not os.path.isdir(contest_dir):
            continue
            
        total_contests += 1
        python_dir = os.path.join(contest_dir, "Python")
        
        if os.path.exists(python_dir):
            python_contests += 1
            
            # Check each program in the Python directory
            for program in os.listdir(python_dir):
                program_dir = os.path.join(python_dir, program)
                if not os.path.isdir(program_dir):
                    continue
                    
                required_files = [
                    "faultyVersion.py",
                    "correctVersion.py",
                    "faultLocation.txt"
                ]
                
                # Check if all required files exist
                has_all_files = all(
                    os.path.exists(os.path.join(program_dir, f))
                    for f in required_files
                )
                
                if has_all_files:
                    python_programs.append({
                        'contest': contest,
                        'program': program,
                        'path': program_dir
                    })
    
    logger.info(f"Found {total_contests} total contests")
    logger.info(f"Found {python_contests} contests with Python programs")
    logger.info(f"Found {len(python_programs)} complete Python programs")
    
    if len(python_programs) == 0:
        logger.error("No valid Python programs found!")
        return False
        
    # Print some example programs
    logger.info("\nExample programs found:")
    for prog in python_programs[:3]:
        logger.info(f"Contest: {prog['contest']}, Program: {prog['program']}")
        
    return python_programs

def prepare_training_data(python_programs, output_dir):
    """
    Prepare and organize the training data
    """
    train_dir = os.path.join(output_dir, "training_data")
    os.makedirs(train_dir, exist_ok=True)
    
    for i, prog in enumerate(python_programs):
        program_dir = os.path.join(train_dir, f"program_{i}")
        os.makedirs(program_dir, exist_ok=True)
        
        # Copy program files
        for file in ["faultyVersion.py", "correctVersion.py", "faultLocation.txt"]:
            src = os.path.join(prog['path'], file)
            dst = os.path.join(program_dir, file)
            shutil.copy2(src, dst)
    
    logger.info(f"\nPrepared {len(python_programs)} programs for training in {train_dir}")
    return train_dir

if __name__ == "__main__":
    code_dir = os.path.join(os.getcwd(), "Code")
    output_dir = os.path.join(os.getcwd(), "fault_localizer_output")
    
    # Check and prepare data
    python_programs = check_and_prepare_data(code_dir)
    if python_programs:
        train_dir = prepare_training_data(python_programs, output_dir)
        logger.info(f"Data preparation complete. Training data in: {train_dir}")
    else:
        logger.error("Data preparation failed!")
