# FASTAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Install dependencies

```sh
pip install fastapi[all]

OR

pip install fastapi "uvicorn[standard]"
```

## Start server
```sh
uvicorn <filename>:<fastapi-object-name> --reload
```
Example:
    If your file name is `main.py` and in this file you make FastAPI object like this: `app = FastAPI()`  
    then the command will look like this:
    ```uvicorn main:app --reload```

