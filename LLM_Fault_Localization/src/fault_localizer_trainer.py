import os
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_directory_structure(base_dir):
    """Verify and print the directory structure"""
    logger.info(f"Checking directory structure in: {base_dir}")
   
    # Check Code directory
    code_dir = os.path.join(base_dir, "Code")
    tool_dir = os.path.join(base_dir, "Tool")
   
    if not os.path.exists(code_dir):
        logger.error(f"Code directory missing at: {code_dir}")
    else:
        logger.info(f"Found Code directory: {code_dir}")
        # List contents of Code directory
        contents = os.listdir(code_dir)
        logger.info(f"Code directory contents: {contents[:5]}...")
   
    if not os.path.exists(tool_dir):
        logger.error(f"Tool directory missing at: {tool_dir}")
    else:
        logger.info(f"Found Tool directory: {tool_dir}")
        # List contents of Tool directory
        contents = os.listdir(tool_dir)
        logger.info(f"Tool directory contents: {contents}")

def load_metadata_files(base_dir):
    """Load date and difficulty information"""
    tool_dir = os.path.join(base_dir, "Tool")
    metadata = {}
   
    logger.info(f"Loading metadata from: {tool_dir}")
   
    try:
        # Load date file
        date_file = os.path.join(tool_dir, "date")
        if os.path.exists(date_file):
            with open(date_file, 'rb') as f:
                metadata['dates'] = pickle.load(f)
                logger.info(f"Successfully loaded date information with {len(metadata['dates'])} entries")
        else:
            logger.warning(f"Date file not found at: {date_file}")
       
        # Load difficulty file
        difficulty_file = os.path.join(tool_dir, "difficulty")
        if os.path.exists(difficulty_file):
            with open(difficulty_file, 'rb') as f:
                metadata['difficulties'] = pickle.load(f)
                logger.info(f"Successfully loaded difficulty information with {len(metadata['difficulties'])} entries")
        else:
            logger.warning(f"Difficulty file not found at: {difficulty_file}")
           
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
   
    return metadata

class FaultLocalizationDataset(Dataset):
    def __init__(self, base_dir, tokenizer, max_length=512):
        """Initialize dataset with base directory path"""
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
       
        # Load metadata
        self.metadata = load_metadata_files(base_dir)
       
        logger.info(f"Loading programs from {base_dir}")
        self.examples = self._load_programs()
        logger.info(f"Loaded {len(self.examples)} examples")

    def _load_programs(self):
        """Load all program examples"""
        examples = []
       
        # Check Code directory
        code_dir = os.path.join(self.base_dir, "Code")
        if not os.path.exists(code_dir):
            logger.error(f"Code directory not found: {code_dir}")
            return examples

        # Process each contest
        contest_dirs = [d for d in os.listdir(code_dir)
                       if os.path.isdir(os.path.join(code_dir, d))]
       
        logger.info(f"Found {len(contest_dirs)} contest directories")
       
        for contest in tqdm(contest_dirs, desc="Processing contests"):
            python_dir = os.path.join(code_dir, contest, "Python")
            if not os.path.exists(python_dir):
                continue
               
            for program in os.listdir(python_dir):
                program_path = os.path.join(python_dir, program)
                if not os.path.isdir(program_path):
                    continue
                   
                try:
                    # Load program files
                    faulty_path = os.path.join(program_path, "faultyVersion.py")
                    fault_loc_path = os.path.join(program_path, "faultLocation.txt")
                   
                    if not os.path.exists(faulty_path) or not os.path.exists(fault_loc_path):
                        continue
                       
                    with open(faulty_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    with open(fault_loc_path, 'r') as f:
                        fault_line = int(f.read().strip())
                       
                    # Get metadata if available
                    difficulty = self.metadata.get('difficulties', {}).get(contest, "Unknown")
                    date = self.metadata.get('dates', {}).get(contest, "Unknown")
                       
                    # Process each line
                    code_lines = code.split('\n')
                    for i, line in enumerate(code_lines, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                           
                        examples.append({
                            'code': code,
                            'line': line,
                            'line_number': i,
                            'is_fault': 1 if i == fault_line else 0,
                            'program': program,
                            'contest': contest,
                            'difficulty': difficulty,
                            'date': date
                        })
                       
                except Exception as e:
                    logger.error(f"Error processing {program_path}: {e}")
                    continue
                   
        return examples

    def __len__(self):
        """Return the number of examples"""
        return len(self.examples)
       
    def __getitem__(self, idx):
        """Get a single example"""
        example = self.examples[idx]
       
        # Create input text with metadata
        input_text = f"""
        Contest: {example['contest']}
        Difficulty: {example['difficulty']}
        Date: {example['date']}
       
        Code:
        {example['code']}
       
        Line Number: {example['line_number']}
        Line: {example['line']}
        """
       
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
       
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(example['is_fault'])
        }

def train_fault_localizer(base_dir, output_dir, model_name="microsoft/codebert-base"):
    """Train the fault localization model"""
    # Check directory structure first
    check_directory_structure(base_dir)
   
    # Initialize tokenizer and model
    logger.info(f"Initializing {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
   
    # Create dataset
    dataset = FaultLocalizationDataset(base_dir, tokenizer)
   
    if len(dataset) == 0:
        raise ValueError("No training examples found! Please check the data directory structure.")
       
    logger.info(f"Dataset created with {len(dataset)} examples")
   
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, len(dataset) - train_size]
    )
   
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        eval_strategy="steps"
    )
   
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
   
    # Train the model
    logger.info("Starting training...")
    trainer.train()
   
    # Save the model
    model_save_path = os.path.join(output_dir, "fault_localizer_final")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
   
    return model, tokenizer

if __name__ == "__main__":
    # Get the ConDefects directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    condefects_dir = current_dir  # Since we're already in the ConDefects directory
   
    logger.info(f"Using ConDefects directory: {condefects_dir}")
   
    # Setup paths
    output_dir = os.path.join(condefects_dir, "fault_localizer_output")
    os.makedirs(output_dir, exist_ok=True)
   
    try:
        # Train model
        model, tokenizer = train_fault_localizer(condefects_dir, output_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please check the directory structure and file paths")
        raise  # Re-raise the exception to see the full traceback