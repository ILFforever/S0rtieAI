# SortieAI
Python based GUI application for sorting and assessing image focus using Yolo11 object detection and PyWavelets wave transform.
![alt text](https://github.com/ILFforever/S0rtieAI/blob/main/image/startscr.png "Start Screen")
![alt text](https://github.com/ILFforever/S0rtieAI/blob/main/image/sample_detect.png "Sample_detect")
**Features Implemented**
- Human and animal subject detection
- Waveform Image focus detection
- Bulk Image processing
- Post sort summary with Graph and Preview
- Ability to upload your own YOLO V11 models (.pt) to detect custom subjects (You will need to update the scoring algorithm)
![alt text](https://github.com/ILFforever/S0rtieAI/blob/main/image/options.png "Options")

**Instructions**
- i Select Photos you wish to sort
- ii Press Start
- iii Wait
- iii Photos are automatically moved (or copied) to the specified folder

(Additional settings can be found in the Options tab)
(Needs Default.pt in root folder as base model unless alternative specified in options)
_Shipped with Default model version 3 but will probably updated later on._

*Default.png in root folder is not required but the program will return Path not found on start-up*

