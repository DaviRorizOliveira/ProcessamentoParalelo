#include <bits/stdc++.h> // Inclui (quase) todas as bibliotecas padrão do C++
#include <omp.h>
#include <cuda_runtime.h>

/*
Comandos para compilar:

cd Trabalho03
nvcc -Xcompiler -fopenmp -O3 -o ai_meu ai_meu.cu
./ai_meu
*/

using namespace std;

constexpr int TILE = 32;

double* gerar_matriz(int N) {
    double* mat = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
    random_device rd;
    // Utiliza o Mersenne Twister, mais rápido que a função rand()
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0); // Intervalo utilizado como 0 e 100
    for (int i = 0; i < N * N; ++i) mat[i] = dis(gen);
    return mat;
}

// Função para multiplicação de matrizes sequencial (ordem ikj)
void dgemm_seq(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N) {
    const int block_size = 128;
    // Divide em blocos de linhas de A
    for (int i = 0 ; i < N ; i += block_size) {
        // Divide em blocos intermediários (colunas de A / linhas de B)
        for (int k = 0 ; k < N ; k += block_size) {
            int i_max = min(i + block_size, N);
            int k_max = min(k + block_size, N);
            // Divide em blocos de colunas de B
            for (int j = 0 ; j < N ; j += block_size) {
                int j_max = min(j + block_size, N);
                for (int ii = i ; ii < i_max ; ++ii) {
                    for (int kk = k ; kk < k_max ; ++kk) {
                        double a_ik = A[ii * N + kk];
                        for (int jj = j ; jj < j_max ; ++jj) {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

// Função para multiplicação de matrizes paralela (ordem ikj) usando OpenMP
void dgemm_par(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N, int num_threads) {
    const int block_size = 128;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0 ; i < N ; i += block_size) {
            for (int k = 0 ; k < N ; k += block_size) {
                int i_max = min(i + block_size, N);
                int k_max = min(k + block_size, N);
                for (int j = 0 ; j < N ; j += block_size) {
                    int j_max = min(j + block_size, N);
                    for (int ii = i ; ii < i_max ; ++ii) {
                        for (int kk = k ; kk < k_max ; ++kk) {
                            double a_ik = A[ii * N + kk];
                            for (int jj = j ; jj < j_max ; ++jj) {
                                C[ii * N + jj] += a_ik * B[kk * N + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}


// Versão CUDA otimizada com memória compartilhada (TILED)
__global__ void dgemm_kernel_tiled(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N) {
    // Memória compartilhada por bloco
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double value = 0.0;

    // Percorre os tiles
    for (int m = 0; m < (N + TILE - 1) / TILE; ++m) {
        int aRow = row;
        int aCol = m * TILE + threadIdx.x;
        int bRow = m * TILE + threadIdx.y;
        int bCol = col;

        // Carrega A em shared (se dentro dos limites)
        if (aRow < N && aCol < N) sA[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else sA[threadIdx.y][threadIdx.x] = 0.0;

        // Carrega B em shared (se dentro dos limites)
        if (bRow < N && bCol < N) sB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads(); // Sincronização

        // Multiplicação do tile atual
        for (int e = 0; e < TILE; ++e) {
            value += sA[threadIdx.y][e] * sB[e][threadIdx.x];
        }
        __syncthreads(); // Sincronização
    }

    // Escreve o resultado em C
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Wrapper da versão otimizada, também mede o tempo
double dgemm_cuda(const double* A_host, const double* B_host, double* C_host, int N) {
    // Cria eventos para medir
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Ponteiros device
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    size_t bytes = (size_t)N * N * sizeof(double);

    // Aloca device
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaEventRecord(start);

    // Copia A e B
    cudaMemcpy(d_A, A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Invoca kernel
    dgemm_kernel_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaGetLastError();

    // Copia resultado de volta
    cudaMemcpy(C_host, d_C, bytes, cudaMemcpyDeviceToHost);

    // Medição do tempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Libera device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)ms / 1000.0;
}

// Versão CUDA básica, 1 thread por elemento (Naive)
__global__ void dgemm_kernel_naive(const double* __restrict__ A, const double* __restrict__ B,  double* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Wrapper da versão básica, também mede o tempo
double dgemm_cuda_naive(const double* A_host, const double* B_host, double* C_host, int N) {
    // Cria eventos para medir
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Ponteiros device
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    size_t bytes = (size_t)N * N * sizeof(double);

    // Aloca device
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEventRecord(start);

    // Copia A e B
    cudaMemcpy(d_A, A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Invoca Kernel
    dgemm_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copia resultado de volta
    cudaMemcpy(C_host, d_C, bytes, cudaMemcpyDeviceToHost);

    // Medição do tempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Libera device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 1000.0;
}

// Função para validar resultado
bool validar_resultado(const double* C_ref, const double* C_test, int N, double& delta_max) {
    const double eps = 1e-12;
    const double tolerancia = 1e-8;
    delta_max = 0.0;

    for (int i = 0 ; i < N * N ; ++i) {
        double diff = fabs(C_ref[i] - C_test[i]);
        double denom = fabs(C_ref[i]) + eps;
        double delta = diff / denom;
        delta_max = max(delta_max, delta);
    }

    return (delta_max <= tolerancia);
}

// Função para medir tempo (sequencial)
double medir_tempo_seq(void (*func)(const double*, const double*, double*, int), const double* A, const double* B, double* C, int N) {
    double start = omp_get_wtime();
    func(A, B, C, N);
    return omp_get_wtime() - start;
}

// Função para medir tempo (paralela com OpenMP)
double medir_tempo_par(void (*func)(const double*, const double*, double*, int, int), const double* A, const double* B, double* C, int N, int threads) {
    double start = omp_get_wtime();
    func(A, B, C, N, threads);
    return omp_get_wtime() - start;
}

int main(int argc, char** argv) {
    ofstream csv;
    csv.open("resultados.csv", ios::app);
    if (!csv.is_open()) {
        cerr << "Erro ao abrir resultados.csv\n";
        return 1;
    }
    
    // Cabeçalho do CSV
    if (csv.tellp() == 0) {
        csv << "tamMatriz,tempoSequencial,"
            "tempo2Thread,speedup2Thread,eficiencia2Thread,delta2Thread,"
            "tempo4Thread,speedup4Thread,eficiencia4Thread,delta4Thread,"
            "tempo8Thread,speedup8Thread,eficiencia8Thread,delta8Thread,"
            "tempoCUDA_naive,speedupCUDA_naive,deltaCUDA_naive,"
            "tempoCUDA_tiled,speedupCUDA_tiled,deltaCUDA_tiled\n";
    }

    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    vector<int> num_threads = {2, 4, 8};

    // Percorre o vetor de tamanhos de matrizes
    for (int N : sizes) {
        // Gera as matrizes A e B
        double* A = gerar_matriz(N);
        double* B = gerar_matriz(N);
        
        // Aloca a matriz C do sequencial
        double* C_seq = nullptr;

        // Sequencial
        cudaMallocHost((void**)&C_seq, N * N * sizeof(double));
        memset(C_seq, 0, N * N * sizeof(double));
        double tempo_seq = medir_tempo_seq(dgemm_seq, A, B, C_seq, N);

        // OpenMP
        vector<double> tempos_omp(num_threads.size());
        vector<double> deltas_omp(num_threads.size());
        
        for (size_t idx = 0; idx < num_threads.size(); ++idx) {
            int nt = num_threads[idx];
            double* C_par = nullptr;
            cudaMallocHost((void**)&C_par, N * N * sizeof(double));
            memset(C_par, 0, N * N * sizeof(double));
            tempos_omp[idx] = medir_tempo_par(dgemm_par, A, B, C_par, N, nt);
            double delta;
            if (!validar_resultado(C_seq, C_par, N, delta)) {
                cerr << "ERRO: Resultado OpenMP (" << nt << " threads) diverge! Delta = " << delta << "\n";
            }
            deltas_omp[idx] = delta;
            cudaFreeHost(C_par);
        }

        // CUDA Naive
        double* C_cuda_naive = nullptr;
        cudaMallocHost((void**)&C_cuda_naive, N * N * sizeof(double));
        memset(C_cuda_naive, 0, N * N * sizeof(double));
        double tempo_naive = dgemm_cuda_naive(A, B, C_cuda_naive, N);
        double delta_naive;
        bool ok_naive = validar_resultado(C_seq, C_cuda_naive, N, delta_naive);
        if (!ok_naive) cerr << "ERRO CUDA NAIVE N=" << N << " delta=" << delta_naive << endl;

        // CUDA Tiled
        double* C_cuda_tiled = nullptr;
        cudaMallocHost((void**)&C_cuda_tiled, N * N * sizeof(double));
        memset(C_cuda_tiled, 0, N * N * sizeof(double));
        double tempo_tiled = dgemm_cuda(A, B, C_cuda_tiled, N);
        double delta_tiled;
        if (!validar_resultado(C_seq, C_cuda_tiled, N, delta_tiled)) {
            cerr << "ERRO: Resultado CUDA diverge! Delta = " << delta_tiled << "\n";
        }

        // Grava no .csv
        csv << N << "," << fixed << setprecision(6) << tempo_seq;
        for (size_t idx = 0; idx < num_threads.size(); ++idx) {
            double t = tempos_omp[idx];
            double sp = tempo_seq / t;
            double ef = sp / num_threads[idx];
            csv << "," << t << "," << sp << "," << ef << "," << scientific << deltas_omp[idx];
        }
        double sp_naive = tempo_seq / tempo_naive;
        csv << "," << fixed << setprecision(6) << tempo_naive
            << "," << sp_naive
            << "," << scientific << delta_naive;
        double sp_tiled = tempo_seq / tempo_tiled;
        csv << "," << fixed << tempo_tiled
            << "," << sp_tiled
            << "," << scientific << delta_tiled << "\n";

        // libera memória
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(C_seq);
        cudaFreeHost(C_cuda_naive);
        cudaFreeHost(C_cuda_tiled);
    }

    // Fecha o arquivo CSV
    csv.close();
    
    return 0;
}