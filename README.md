# Training Templates: A collection of machine learning training workflows
The training_templates library provides a collection of classes that execute common training workflows. It is meant to be used as an example of DRY code development and unit testing. The example workflows utilize hyperopt for hyperparameter tuning and log all parameters, fit statistics, and trained models to an MLflow Experiment. Base model workflow classes can be extended to add additional functionality or overridden to alter existing functionality.  

This library was developed using the Databricks extension for VS Code. The extension allows local code to be synced to a Databricks Repo and executed on Databricks compute. The project was developed using a combination of local unit tests and interactive experimentation in Databricks by importing functions and classes into Notebooks. Commits to github were executed from the Repo.

### To run the code locally:

1. Clone the repository.
2. Create a local conda environment.
```
conda create -n training_templates python=3.9
conda activate training_templates
```  
3. If JDK is not installed on your local machine, install it.  
```
conda install -c conda-forge openjdk=11.0.15
```

4. Install the package locally.  
```
pip install -e ".[local,test]"
```
5. Optionally create a Databricks Repo, open this project in VS Code, and sync the local code to the Databricks Repo using the VS Code extension.

6. To run tests, execute the below line.
```
pytest -v --cov
```

### To run the code exclusively in Databricks
1. Clone the repository into a Databricks Repo and open a notebook from the examples directory to see usage examples. 


This project leverages github actions to execute workflows triggered by pull requests and the creation of releases. Contributors to the project create feature branches to develop new functionality. These updates are then added to the master branch through a pull request. The request triggers a github actions workflow that checks the code style and executes unit tests. If either of these fail, the pull request will not be merged.  

Once the master branch has accumulated enough updates to release a new version of the library, a new github release is created, which triggers the second github actions workflow. This workflow accepts the release information from github, builds a new version of the package with that information, and pushes a wheel file to an artifact repository (gemfury). Databrick users can then pip install the library. See the examples directory for details on installing and using the library on a Databricks cluster.



