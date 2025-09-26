#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

/*
Comandos para compilar:

cd Trabalho01
g++ -o teste -Wall -O3 -fopenmp aprimorado.cpp
./teste
*/

// Função para gerar matriz aleatória
unique_ptr<double[]> gerar_matriz(int N) {
    auto mat = unique_ptr<double[]>(new double[N * N]);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dis(gen);
    }
    return mat;
}

// Função para multiplicação de matrizes sequencial (ordem ikj)

/*
A Função divide as matrizes em blocos de tamanho block_size x block_size para melhorar a localidade de referência e o uso da cache.
Se o valor de N não for múltiplo de block_size, o uso de min garante que os loops internos não ultrapassem os limites da matriz.
Se o valor de N for menor que block_size, a função ainda funciona corretamente, processando a matriz inteira de uma vez, sem utilizar o bloqueamento.
A ordem dos loops é i (linhas de C), k (acumulação), j (colunas de C), permitindo acessos contíguos a B e C.
A variável a_ik armazena temporariamente o valor de A[ii * N + kk] para reduzir acessos à memória.
*/
void dgemm_seq(const double* A, const double* B, double* C, int N) {
    const int block_size = 128; // Ajuste conforme o hardware
    // Divide em blocos de linhas
    for (int i = 0; i < N; i += block_size) {
        int i_max = min(i + block_size, N);
        // Divide em blocos intermediários
        for (int k = 0; k < N; k += block_size) {
            int k_max = min(k + block_size, N);
            // Divide em blocos de colunas
            for (int j = 0; j < N; j += block_size) {
                int j_max = min(j + block_size, N);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int kk = k; kk < k_max; ++kk) {
                        double a_ik = A[ii * N + kk]; // Cache A
                        for (int jj = j; jj < j_max; ++jj) {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

// Função para multiplicação de matrizes paralela (ordem ikj)
void dgemm_par(const double* A, const double* B, double* C, int N, int num_threads) {
    const int block_size = 128; // Ajuste conforme o hardware
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for
        // Divide em blocos de linhas
        for (int i = 0; i < N; i += block_size) {
            // Divide em blocos intermediários
            for (int k = 0; k < N; k += block_size) {
                int i_max = min(i + block_size, N);
                int k_max = min(k + block_size, N);
                // Divide em blocos de colunas
                for (int j = 0; j < N; j += block_size) {
                    int j_max = min(j + block_size, N);
                    for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                            double a_ik = A[ii * N + kk]; // Cache A
                            #pragma omp simd // Força vetorização
                            for (int jj = j; jj < j_max; ++jj) {
                                C[ii * N + jj] += a_ik * B[kk * N + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Função para medir tempo (sequencial)
double medir_tempo(void (*dgemm_func)(const double*, const double*, double*, int), const double* A, const double* B, double* C, int N) {
    auto start = chrono::high_resolution_clock::now();
    dgemm_func(A, B, C, N);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// Função para medir tempo (paralela), usa sobrecarga de função
double medir_tempo(void (*dgemm_func)(const double*, const double*, double*, int, int), const double* A, const double* B, double* C, int N, int num_threads) {
    auto start = chrono::high_resolution_clock::now();
    dgemm_func(A, B, C, N, num_threads);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() {
    /*
    Acelera entrada e saída (cin/cout), não há entrada, mas há saída
    Não muda o tempo de execução das funções de multiplicação, somente a parte de In/Out
    */
    ios_base::sync_with_stdio(false);

    // Mudar esse vector, pouco otimizado
    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    vector<int> num_threads = {2, 4, 8};

    // Percorre o vetor de tamanhos de matrizes
    for (int N : sizes) {
        cout << "Tamanho da matriz: " << N << "x" << N << endl;

        // Gerar matrizes A e B
        auto A = gerar_matriz(N);
        auto B = gerar_matriz(N);
        
        // Gera matrizes C com zeros
        auto C_seq = unique_ptr<double[]>(new double[N * N]()); // Inicializa com zeros
        auto C_par = unique_ptr<double[]>(new double[N * N]());

        // Versão sequencial
        double tempo_seq = medir_tempo(dgemm_seq, A.get(), B.get(), C_seq.get(), N);
        cout << "Tempo sequencial: " << tempo_seq << " segundos" << endl;

        // Versão paralela
        for (int threads : num_threads) {
            fill(C_par.get(), C_par.get() + N * N, 0.0); // Reinicializa C_par com zeros
            double tempo_par = medir_tempo(dgemm_par, A.get(), B.get(), C_par.get(), N, threads);
            cout << "Tempo paralelo com " << threads << " threads: " << tempo_par << " segundos" << endl;

            // Cálculo de métricas
            double speedup = tempo_seq / tempo_par;
            double eficiencia = speedup / threads; // Tratando eficiência como speedup dividido pelo número de threads: Confirmar com o professor se está correto
            cout << "Speedup: " << speedup << endl;
            cout << "Eficiência: " << eficiencia << endl;
        }

        cout << endl;
    }

    return 0;
}