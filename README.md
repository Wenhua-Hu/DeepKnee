## University of Amsterdam AI / XAI project for Medical Organization Quin
- conda create --name DeepKnee python=3.8.8
- conda activate DeepKnee
- git clone https://github.com/whu-linuxer/DeepKnee.git
- cd DeepKnee/apps/data/models && mkdir models
- download the models into directory: models/
- change the model name in correspondence with the names in apps/config.py
- pip install -r requirements.txt
- python run.py

Group Members: Lukas, Jordy, Anna and Wenhua

Contact: [Wenhua Hu](w.hu1224@gmail.com)

[Models and Datasets](https://www.dropbox.com/sh/01pks0pdugpgh07/AAAmRcqFt-BhdqEFIRj3mNkVa?dl=0)
[Movie introduction](https://youtu.be/ul8kfml8PeY)

GUI Functionalities:

- support to search for patient from system - Here we use the local lightweighted database sqllites
- support to update patient personal information and radiographs data, the latest radiographs are ahead of the other radiographs (clinician can focus on the new radiographs being received)
- support to do prediction and explainable analysis in parallel
- support to zoom in the radiographs
- provide thumbnails to hint the progress of processing (HEATMAP, BOUNDINGBOX AND LIME)
- support to choose 6 kinds of single models
- support to switch among radiographs for comparison
- provide different channels for both left and right knees 
- support to give confidence score for 5 grades
- support to give a feedback / decison on top of a specific XAI image (Clicking on the thumbnail to switch the corresponding comment box, Quin could evaluate or improve the model on top of these data)
- support to show the metrics on top of each model being selected (the question mark can show the desc on hovering)

Notes: the Lime needs some minutes to analysis

Inference:

![Inference of DeepKnee](./apps/data/examples/inference.png)

Metrics:

![Metrics of DeepKnee](./apps/data/examples/metrics.png)
