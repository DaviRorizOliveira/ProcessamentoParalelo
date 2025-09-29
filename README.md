# Processamento Paralelo

Este repositório é destinado aos projetos realizados na disciplina DEC-107 Processamento Paralelo.
<br>Estes Projetos descritos foram produzidos por:

[Davi Roriz Oliveira](https://github.com/DaviRorizOliveira)

[Estêvão Sousa Vieira](https://github.com/ESTEV40)

## Projeto 1: DGEMM - Multiplicação de Matrizes com OpenMP

Este projeto implementa a multiplicação de matrizes (DGEMM) em versões sequencial e paralela utilizando OpenMP. O objetivo é comparar o desempenho entre as duas abordagens em diferentes tamanhos de matrizes e números de threads.

### Requisitos

- Compilador C++ com suporte a OpenMP (g++ ou clang++)
- Python 3.12+ com as bibliotecas: pandas, matplotlib

### Compilação, execução e visualização

#### Método 1: (C++ e Python)

Caso ainda não tenha compilado o main.cpp
```bash
g++ -o main -Wall -O3 -fopenmp -march=native -mfma main.cpp
```
Execute o arquivo main
```bash
./main
```
Posteriormente, execute o `arquivo analise.py`, para visualização dos dados.
```bash
python3 analisador.py
```

#### Método 2: (python)
Execute o arquivo `run.py`, irá fazer todo processo automatizado.
```bash
python3 run.py
```
### [Relatório do Projeto 1](https://docs.google.com/document/d/1PRxJCbYw2ydfvzLtNcZFzmeyUYP9l9qnrQP6-961W_4/edit?usp=sharing)
