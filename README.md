# J-POD

## Installation
To install the `j-pod` pipeline and make the `j-pod` command available globally in your terminal, follow these steps:

### 1. Clone the repo
First, clone the project repository:
```bash
git clone https://github.com/your-username/j-pod.git
```
### 2. Navigate to the project directory
```bash
cd j-pod
```
### 3. Install the package
Install the package in editable mode using `pip`. The `-e` flag ensures that any changes you make to the source code are immediately reflected without needing to reinstall:

(*Note: It is highly recommended to run this inside a Python virtual environment*)
```bash
pip install -e .
```
### 4. Verify the installation
Once the installation is complete, you can verify that the CLI command is registered and working by checking the help menu:
```bash
j-pod --help
```

### 5. Have fun with the pipeline
To use the pipeline you can proceed in two ways:
1. Move to your project/working directory and run the command from there. The package will use the current directory as root directory for the `data` and `output` directories.
2. You can run the code wherever you want. Just remember to specify the project/working directory using `-w`/`--workdir` argument. For the other directories, just specify the relative path.

**Note:** The default directory for the FITS files is `./data`. The default directory for the output files is `./output`; if `output`, it will be created automatically.

