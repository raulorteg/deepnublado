## Code formatting
----------------------

We use ```black``` for code formatting and ```isort``` for sorting the imports.
To use them first install the following versions:

```bash
python -m pip install black==22.10.0 isort==5.10.1
```

Use them on the src directory (at ```deepnublado/src```):

```bash
python -m black .
python -m isort .
```