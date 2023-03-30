# Training Templates: A collection of training workflows for common ML libraries.

### Local environment setup  
1. Create a local conda environment
```
conda create -n training_templates python=3.9
conda activate training_templates
```  

2. If JDK is not installed on your local machine, install it  
```
conda install -c conda-forge openjdk=11.0.15
```

3. Install the package locall  
```
pip install -e ".[local,test]"
```



