# cs441_final_project_LLM_fault_localization

Neural Network-based Fault Localization using CodeBERT
This project implements a neural network-based fault localization system using CodeBERT. The system learns from faulty code examples in the ConDefects dataset to predict potential fault locations in Python code.
Architecture Overview
The system uses a fine-tuned CodeBERT model with a classification head to identify faulty lines of code. The architecture consists of:

Base Model:

CodeBERT (microsoft/codebert-base)
12 transformer encoder layers
768 hidden dimensions
12 attention heads


Classification Layer:

Binary classification (fault/non-fault)
Dense layer followed by output projection
Softmax activation for probability output



Dataset Structure
The system uses the ConDefects dataset with the following structure:
CopyConDefects/
├── Code/
│   └── [contest_folders]/
│       └── Python/
│           └── [program_folders]/
│               ├── faultyVersion.py
│               ├── correctVersion.py
│               └── faultLocation.txt
├── Tool/
│   ├── date
│   └── difficulty
└── fault_localizer_output/
    └── [trained_model_files]
Setup and Installation

Install required dependencies:

bashCopypip install torch transformers datasets scikit-learn numpy pandas tqdm accelerate

Clone the ConDefects repository:

bashCopygit clone https://github.com/appmlk/ConDefects.git
cd ConDefects

Make sure you have the binary metadata files in place:

Tool/date: Contains program date information
Tool/difficulty: Contains program difficulty information



Training Process
The training process follows these steps:

Data Preparation:

Loads Python programs from the Code directory
Processes fault location information
Integrates metadata (dates and difficulties)
Creates training examples for each line of code


Input Processing:
pythonCopyinput_text = f"""
Contest: {contest}
Difficulty: {difficulty}
Date: {date}
Code: {code}
Line Number: {line_number}
Line: {line}
"""

Training Configuration:

Batch size: 8
Learning rate with warmup
Weight decay: 0.01
3 epochs
80/20 train/validation split



Usage

Train the model:

bashCopypython fault_localizer_trainer.py

The trained model will be saved in fault_localizer_output/

Model Training Details
The training process includes:

Data Loading:

Processes Python programs from ConDefects dataset
Extracts fault locations from annotations
Incorporates program metadata


Input Features:

Program code context
Line number information
Program metadata (contest, difficulty, date)
Line content


Training Strategy:

Binary classification at line level
Uses negative sampling for non-faulty lines
Incorporates program metadata for context



Training Parameters
pythonCopytraining_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=500,
    save_steps=1000
)
Methodology

Data Processing:

Each program is split into individual lines
Fault locations are used as positive examples
Non-faulty lines serve as negative examples
Program metadata provides additional context


Model Architecture:

CodeBERT base model for code understanding
Classification head for fault prediction
Contextual embeddings for code representation


Training Approach:

Fine-tuning of pre-trained CodeBERT
Binary classification training
Evaluation on hold-out validation set



Directory Structure
Copyproject/
├── fault_localizer_trainer.py  # Main training script
├── main.py                     # Main execution script
├── README.md                   # This file
└── requirements.txt           # Dependencies
Contributing

Fork the repository
Create your feature branch
Commit your changes
Push to the branch
Create a new Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

CodeBERT: https://github.com/microsoft/CodeBERT
ConDefects Dataset: https://github.com/appmlk/ConDefects