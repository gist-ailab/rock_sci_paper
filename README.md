# rock_sci_paper
**AIlab Project for GSH - GIST Creative Research Program**<br />
**Train Classifier for Rock Scissor Paper Hand gesture using deep learning**<br />

## Code Overview
1. **data_generation.py** : Generating rock-scissor-paper hand gesture image dataset by taking a photo of gesture in order to corresponding label<br />
2. **dataset.py** : Loader of rock-scissor-paper dataset<br />
3. **model.py** : Resnet model but user can access to output feature of each blocks<br />
4. **train_teacher.ipynb** : Train teacher model<br />
5. **train_student.ipynb** : Train student model by distilating feature of teacher model<br />
6. **CAM.ipynb** : Visualize CAM of trained model<br />
7. **inference.ipynb** : Real time Rock-Scissor-Paper inference using webcam
