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
   Download the pretrained weights and parameters from the following link: https://drive.google.com/drive/folders/1R_EsvjougYKuMO2S44ArEI4xgYWuJ-bD?usp=sharing
   These files include:

   - 32-bit floating point weights.
   - Weights with different quantization types used in the experiments of the paper.

3. **Place Weights in the Corresponding Folders**:   
   After downloading, place each file in its corresponding folder as shown below:   

   ```bash
   ├── VGG-11/
   │   ├── models/
   │   ├── state-dicts/
   │   ├── quantized_type1/
   │   ├── quantized_type2/
   │   └── ...



4. **Run the Scripts**:   
   Run the q.....py files to view the results.  These scripts will use the weights and parameters and output the results of the experiments.   

```bash
python q_example.py

5. **View Results**:   
   The script outputs the experiment results in the console.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

