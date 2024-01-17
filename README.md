# Auto-Defense-ML Project Setup

Follow these steps to set up the Auto-Defense-ML project on both Windows and Linux environments:

## Step 1: Clone Repositories
Clone the main Auto-Defense-ML repository and the forked adversarial-robustness-toolbox repository by running the following commands:

```bash
git clone https://github.com/RonVol/Auto-Defense-ML.git
git clone https://github.com/RonVol/adversarial-robustness-toolbox.git
```
## Step 2: Clone Repositories
Navigate to the Auto-Defense-ML project directory and create a virtual environment.<br>
Select the Python interpreter version 3.11.7 and activate the virtual environment.
### For Windows:
```bash
cd Auto-Defense-ML
python -m venv venv
.\venv\Scripts\activate
```
### For Linux:
```bash
cd Auto-Defense-ML
python3 -m venv venv
source venv/bin/activate
```
## Step 3: Install Forked Repository Locally
To install the forked adversarial-robustness-toolbox repository, you have two options: with and without the -e flag.<br>
The -e flag in pip install -e /path/to/adversarial-robustness-toolbox is used to install a package in "editable" mode.<br> When you install a package with the -e flag, it creates a symlink or a .pth file, allowing you to actively develop the package and have the changes immediately reflected in your project without the need to reinstall the package each time you make a modification.
```bash
pip install -e /path/to/adversarial-robustness-toolbox
```
## Step 4: Install Dependencies from requirements.txt
Install the project dependencies using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```



