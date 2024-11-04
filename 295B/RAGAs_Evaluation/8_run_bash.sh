#!/bin/bash
source /kapil_wanaskar/295B/RAGAs_Evaluation/2_venv_deepaval/bin/activate

# set environment variables
export DEEPEVAL_RESULTS_FOLDER="/kapil_wanaskar/295B/RAGAs_Evaluation/DEEPEVAL_RESULTS_FOLDER"

# run deepeval test command
deepeval test run /kapil_wanaskar/295B/RAGAs_Evaluation/test_13_Eval_pipeline.py

# call python script to convert json to csv
python3 /kapil_wanaskar/295B/RAGAs_Evaluation/11_json_to_5cols.py