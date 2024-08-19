# rtfMRI-NF

## Setup - create a virtual environment

### Conda + Ubuntu 20.04 (our environment)

```bash
$ conda env create -f conda_requirements.yml
``` 

### Conda (other platforms)

```bash
$ conda env create -f cross_requirements.yml
``` 

## How to use

```bash
$ conda activate ISL_rtNF
$ python main.py
```

## How to edit the UI

1. Edite the `main.ui` file.

2. Run the `pyside6-uic` tool to update the Python file `ui_main.py` from the `main.ui` file ([REF](https://doc.qt.io/qtforpython-6/tutorials/basictutorial/uifiles.html)).

```bash
$ pyside6-uic main.ui -o ui_main.py
```