## Things changed from original
pipelines/aviation/preprocess.py - Added data cycler and changed features corresponding to our project
pipelines/aviation/pipeline.py - Changed all to aviation-related names
pipelines/aviation - Changed folder name from abalone to aviation

## Workflow
![Workflow] (AWS MLOps.png "Workflow")

## Steps to use
1. Create a project using the MLOps template for model building, training, and deployment
2. Clone the repository and replace the above files with those from this repo
3. Upload aviation_main.csv to a location in S3
4. Change line 138 to reflect the location of the uploaded aviation_main.csv
5. Use the notebook Used for Inference.ipynb to make inferences
