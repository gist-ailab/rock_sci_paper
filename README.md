# rock_sci_paper
AIlab Project for GSH - GIST Creative Research Program<br />
Train Classifier for Rock Scissor Paper Hand gesture using deep learning<br />

## Code Overview
data_generation.py : Generating rock-scissor-paper hand gesture image dataset by taking a photo of gesture in order to corresponding label<br />
dataset.py : Loader of rock-scissor-paper dataset<br />
delete.py : Remove all datas in the folder<br />
lowResolutionGen.py : Generating low resolution images of dataset in given path<br />
train.py : Basic Train codes for rock-scissor-paper dataset<br />

## ToDo
1. teacher_train.py : Train teacher model
2. student_train.py : Train student model by distilating feature of teacher model
3. model.py : Resnet model but user can access to output feature of each blocks
4. competiton.py : Evaluation codes that classify given hand gesture and return gesture that can win
5. Class_Lecture.ppt : Presentation for lecture class about basic deep learning and classification and knowledge distilation
6. Lab_introduction.ppt : Presentation for introducing AI lab
