import subprocess

def compilaExecuta() : # caso queira executar o main.cpp diretamente do python
    compile_command = ["g++", "-o", "main", "-Wall", "-O3", "-fopenmp", "main.cpp"]
    exec_command = ["python3", "analise.py"]
    result = subprocess.run(compile_command, capture_output=True, text = True)

    if result.returncode != 0 :
        print(result.stderr)
        exit(1)

    executable = "./main"
    result = subprocess.run(executable, capture_output=True, text = True)
    if result.returncode == 0 :
        subprocess.run(exec_command,capture_output=True, text = True)

def main() :
    compilaExecuta()

if __name__ == "__main__" :
    main()