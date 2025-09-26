import pandas as pd
import numpy as np
import matplotlib as plt
import subprocess

def compilaExecuta() :
    compile_command = ["g++", "-o", "aprimorado", "-Wall", "-O3", "-fopenmp", "aprimorado.cpp"]
    result = subprocess.run(compile_command, capture_output=True, text = True)

    if result.returncode != 0 :
        print(result.stderr)
        exit(1)

    executable = "./aprimorado"
    result = subprocess.run(executable, capture_output=True, text = True)

def analiseDados():
    pass

def main ():
    compilaExecuta()
    analiseDados()

if __name__ == "__main__":
    main()



