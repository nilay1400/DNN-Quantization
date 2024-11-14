# FORTUNE: A Negative Memory Overhead Hardware-Agnostic Fault TOleRance TechniqUe in DNNs

Link to the paper: .....................

This is the official repository for the implementation of FORTUNE: A Negative Memory Overhead Hardware-Agnostic Fault TOleRance TechniqUe in DNNs. This paper presents a hardware-agnostic fault tolerance technique for DNNs that leverages quantization to enhance reliability without significant performance overhead.

## Features
- Negative overhead fault tolerance technique
- Design space exploration framework
- Introduction of two new metrics for evaluating reliability
- Validation


## Usage

1. **Install Required Packages**:  
   First, ensure all required packages are installed. You can do this by running:

   ```bash
   pip install -r requirements.txt


2. **Download Pretrained Weights**:    
   Download the pretrained weights and parameters from the following link: https://drive.google.com/drive/folders/1R_EsvjougYKuMO2S44ArEI4xgYWuJ-bD?usp=sharing.    
   These files include:

   - 32-bit floating point weights.
   - Weights with different quantization types used in the experiments of the paper.

3. **Place Weights in the Corresponding Folders**:   
   After downloading, place each file in its corresponding folder as shown below:
   (The path is shown for 3-bit VGG-11 as an example.)   

   ```bash
   ├── VGG-11/
   │   └── 3-bit/
   │       ├── models/
   │       │   └── state_dicts/
   │       │       ├── vgg11_bn.pt/
   │       ├── qvgg11-0-7.pth/




5. **Run the Scripts**:   
   Run the q0-example.py files to view the results.  These scripts will use the weights and parameters and output the results of the experiments. 'example' should be replaced by the appropriate name in each folder.     

   ```bash
   python3 q0-example.py

6. **View Results**:   
   The script outputs the experiment results in the console.

## Bibtex


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

