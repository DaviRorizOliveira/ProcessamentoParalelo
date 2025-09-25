#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

// Função para multiplicação de matrizes sequencial
void dgemm_seq(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Função para multiplicação de matrizes paralela
void dgemm_par(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// Função para gerar matriz aleatória
vector<double> gerar_matriz(int N) {
    vector<double> mat(N * N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dis(gen);
    }
    return mat;
}

// Função para medir tempo e executar a multiplicação
double medir_tempo(void (*dgemm_func)(const vector<double>&, const vector<double>&, vector<double>&, int), const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    auto start = chrono::high_resolution_clock::now();
    dgemm_func(A, B, C, N);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// Função sobrecarregada para a versão paralela que recebe 'num_threads'
double medir_tempo(void (*dgemm_func)(const vector<double>&, const vector<double>&, vector<double>&, int, int), const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int num_threads) {
    auto start = chrono::high_resolution_clock::now();
    dgemm_func(A, B, C, N, num_threads);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() {
    vector<int> sizes = {512, 1024, 2048, 4096};
    vector<int> num_threads = {2, 4};

    // Multiplicação para cada tamanho de matriz (512, 1024, 2048, 4096)
    for (int N : sizes) {
        cout << "Tamanho da matriz: " << N << "x" << N << endl;

        // Gerar matrizes A e B
        auto A = gerar_matriz(N);
        auto B = gerar_matriz(N);
        
        // Gera a matriz C com zeros
        auto C_seq = vector<double>(N * N, 0.0);
        auto C_par = vector<double>(N * N, 0.0);

        // Versão sequencial
        double tempo_seq = medir_tempo(dgemm_seq, A, B, C_seq, N);
        cout << "Tempo sequencial: " << tempo_seq << " segundos" << endl;

        // Versão paralela para diferentes números de threads (2 e 4)
        for (int threads : num_threads) {
            double tempo_par = medir_tempo(dgemm_par, A, B, C_par, N, threads);
            cout << "Tempo paralelo com " << threads << " threads: " << tempo_par << " segundos" << endl;

            // Cálculo de métricas
            double speedup = tempo_seq / tempo_par;
            double eficiencia = speedup / threads;
            cout << "Speedup: " << speedup << endl;
            cout << "Eficiência: " << eficiencia << endl;
        }

        cout << endl;
    }

    return 0;
}