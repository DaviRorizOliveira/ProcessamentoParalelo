#include <bits/stdc++.h> // Inclui (quase) todas as bibliotecas padrão do C++
#include <omp.h>

using namespace std;

/*
Comandos para compilar:

cd Trabalho01
g++ -o main -Wall -O3 -fopenmp -march=native -mfma main.cpp
./main
*/

// Função para gerar matriz aleatória com valores entre 0 e 100
double* gerar_matriz(int N) {
    double* mat = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
    random_device rd;
    // Utiliza o Mersenne Twister, mais rápido que a função rand()
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0); // Intervalo utilizado como 0 e 100
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dis(gen);
    }
    return mat;
}

// Função para multiplicação de matrizes sequencial (ordem ikj)
void dgemm_seq(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N) {
    const int block_size = 128;
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
                        double a_ik = A[ii * N + kk];
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
void dgemm_par(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N, int num_threads) {
    const int block_size = 128;
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
                            double a_ik = A[ii * N + kk];
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
double medir_tempo_seq(void (*dgemm_func)(const double*, const double*, double*, int), const double* A, const double* B, double* C, int N) {
    double start = omp_get_wtime();
    dgemm_func(A, B, C, N);
    double end = omp_get_wtime();
    return end - start;
}

// Função para medir tempo (paralela)
double medir_tempo_par(void (*dgemm_func)(const double*, const double*, double*, int, int), const double* A, const double* B, double* C, int N, int num_threads) {
    double start = omp_get_wtime();
    dgemm_func(A, B, C, N, num_threads);
    double end = omp_get_wtime();
    return end - start;
}

int main() {
    ofstream csv_file("resultados.csv");
    if (!csv_file.is_open()) {
        cerr << "Erro ao abrir arquivo resultados.csv" << endl;
        return 1;
    }

    csv_file << "tamMatriz,tempoSequencial,tempo2Thread,speedup2Thread,eficiencia2Thread,tempo4Thread,speedup4Thread,eficiencia4Thread,tempo8Thread,speedup8Thread,eficiencia8Thread" << endl;

    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    vector<int> num_threads = {2, 4, 8};

    // Percorre o vetor de tamanhos de matrizes
    for (int N : sizes) {
        csv_file << N << ",";

        // Gerar matrizes A e B
        double* A = gerar_matriz(N);
        double* B = gerar_matriz(N);
        
        // Aloca matriz C
        double* C_seq = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
        double* C_par = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));

        // Versão sequencial
        memset(C_seq, 0, N * N * sizeof(double)); // Inicializa C_seq com zeros
        double tempo_seq = medir_tempo_seq(dgemm_seq, A, B, C_seq, N);
        
        csv_file << tempo_seq;

        // Versão paralela
        for (int threads : num_threads) {
            memset(C_par, 0, N * N * sizeof(double)); // Reinicializa C_par com zeros
            double tempo_par = medir_tempo_par(dgemm_par, A, B, C_par, N, threads);

            // Cálculo de métricas
            double speedup = tempo_seq / tempo_par;
            double eficiencia = speedup / threads;
            
            csv_file << "," << tempo_par << "," << speedup << "," << eficiencia;
        }

        csv_file << endl;
    }

    return 0;
}