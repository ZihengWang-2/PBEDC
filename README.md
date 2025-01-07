# PBEDC

This repository provides the necessary code to perform the simulations for the paper **Perturbation-Based Error Detection and Correction (PBEDC) in Dependable Large-Scale Machine Learning Systems**. 

### **Reference Links**
This repository integrates OpenAI’s CLIP model to evaluate fault injection and detection techniques.
- **CLIP Blog**: [OpenAI Blog](https://openai.com/blog/clip/)
- **CLIP Paper**: [CLIP: Connecting Text and Images](https://arxiv.org/abs/2103.00020)
- **CLIP Colab Notebook**: [Colab Notebook](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

---

## **Contents**

1. **PBEDC Fault Injection**: Simulates fault injection into neural network weights and evaluates detection and correction capabilities.
2. **CLIP Integration**: Loads and utilizes CLIP models for testing using datasets such as CIFAR-10, CIFAR-100, and mini-ImageNet.
3. **PBEDC Sample Selection**:
   - Greedy Search Strategy
   - Max Overlapped Coverage Strategy
4. **Utilities**: Includes progress bars, floating-point bit manipulation, and data processing utilities.

---

## **File Structure**
- `clip.py`: Core CLIP model loading and preprocessing utilities.
- `main.py`: Main execution script for PBEDC fault injection, inference, and sample selection.
- `model.py`: Implementation of neural network architectures including ResNet and Vision Transformers.
- `simple_tokenizer.py`: A simplified tokenizer for text preprocessing using Byte Pair Encoding (BPE).
- `utils.py`: Helper functions for floating-point manipulation, random seed setting, and PBEDC sample selection.

---

## **Dependencies**

The code requires **Python 3.7 or above** and the following libraries:
- `torch` (PyTorch)
- `numpy`
- `pandas`
- `matplotlib`
- `pickle`
- `Pillow` (for image processing)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## **Compilation and Running Instructions**
### **Compilation**
1. Ensure all required libraries are installed.
2. Place all datasets (CIFAR10, CIFAR100, or mini-ImageNet) in the appropriate directories. These can also be downloaded using PyTorch utilities.
### **Running**
Execute the following to run fault injection and PBEDC simulations:
```bash
python main.py
```
### **PBEDC Sample Selection**
To perform PBEDC sample selection:
1. Run the fault injection and detection simulations to generate the required data.
2. Use `find_samples_greedy` or `find_samples_max_coverage` from `utils.py` to select samples.

## **Results**
The results of the simulations are saved in CSV and Excel formats under the output directory. These include:
### **Detection Results**
- `output.xlsx`: Contains detection metrics such as changes prediction and the critical node (softmax).

### **PBEDC Sample Selection**
- `selected_samples.csv`: Details of PBEDC samples selected using Greedy Search and Max Overlapped Coverage strategies.

## **License**
This repository is licensed under the MIT License.
