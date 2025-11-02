#include <bits/stdc++.h> // Inclui (quase) todas as bibliotecas padrão do C++
#include <omp.h>
#include <mpi.h>

/*
Comandos para compilar:

cd Trabalho02
mpic++ -fopenmp -O3 -march=native -mfma -o main main.cpp
mpirun -np 2 ./main
mpirun -np 4 ./main
mpirun -np 8 ./main

Para executar com 8 threads sem ter 8 núcleos (Pode dar problema de memória RAM, então retire o N = 4096):
mpirun -np 8 --oversubscribe ./main
*/

using namespace std;

// Função para gerar matriz aleatória com valores entre 0 e 100
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
}

// Função para multiplicação de matrizes paralela (ordem ikj) usando MPI
void dgemm_mpi(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N, int size, int rank) {
    const int block_size = 128;

    /*
    Exemplo: N = 1024, size = 4
    Cada processo recebe 256 linhas de A
    rank 0: linhas 0 a 255
    rank 1: linhas 256 a 511
    rank 2: linhas 512 a 767
    rank 3: linhas 768 a 1023
    */

    int linhas_por_processo = N / size; // Número de linhas por processo (assumindo N divisível por size, todos recebem igual)

    // Alocação local
    double* local_A = static_cast<double*>(aligned_alloc(64, linhas_por_processo * N * sizeof(double)));
    double* local_C = static_cast<double*>(aligned_alloc(64, linhas_por_processo * N * sizeof(double)));
    memset(local_C, 0, linhas_por_processo * N * sizeof(double));

    /*
    Distribuição com MPI_Scatter
    O processo raiz (rank 0) envia partes de A para todos os processos, incluindo ele mesmo
    Cada processo recebe linhas_por_processo * N elementos
    
    Parâmetros:
        Buffer de envio (A no rank 0, ignorado nos outros)
        Número de elementos enviados para cada processo (linhas_por_processo * N)
        Tipo dos dados enviados
        Buffer de recebimento (local_A em cada processo)
        Número de elementos recebidos por cada processo (linhas_por_processo * N)
        Tipo dos dados recebidos
        Rank do processo que envia os dados (0 neste caso), processo raiz
        Comunicador (MPI_COMM_WORLD para todos os processos)
    */
    MPI_Scatter(A, linhas_por_processo * N, MPI_DOUBLE, local_A, linhas_por_processo * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);             // comunicador
    
    // Divide em blocos de linhas de local_A
    for (int i = 0; i < linhas_por_processo; i += block_size) {
        // Divide em blocos intermediários (colunas de A / linhas de B)
        for (int k = 0; k < N; k += block_size) {
            int i_max = min(i + block_size, linhas_por_processo);
            int k_max = min(k + block_size, N);
            // Divide em blocos de colunas de B
            for (int j = 0; j < N; j += block_size) {
                int j_max = min(j + block_size, N);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int kk = k; kk < k_max; ++kk) {
                        double a_ik = local_A[ii * N + kk];
                        for (int jj = j; jj < j_max; ++jj) {
                            local_C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
    
    /*
    Coleta os resultados com MPI_Gather
    O processo raiz (rank 0) coleta as partes de C de todos os processos, incluindo ele mesmo

    Parâmetros:
        Buffer de envio (local_C em cada processo)
        Número de elementos enviados por cada processo (linhas_por_processo * N)
        Tipo dos dados enviados
        Buffer de recebimento (C no rank 0, ignorado nos outros)
        Número de elementos recebidos de cada processo (linhas_por_processo * N)
        Tipo dos dados recebidos
        Rank do processo que coleta os dados (0 neste caso), processo raiz
        Comunicador (MPI_COMM_WORLD para todos os processos)
    */
    MPI_Gather(local_C, linhas_por_processo * N, MPI_DOUBLE, C, linhas_por_processo * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Libera memória
    free(local_A);
    free(local_C);
}

// Função para validar resultado
bool validar_resultado(const double* C_ref, const double* C_test, int N, double& delta_max) {
    const double eps = 1e-12;
    const double tolerancia = 1e-9;
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

// Função para medir tempo (paralela com MPI)
double medir_tempo_mpi(void (*func)(const double*, const double*, double*, int, int, int), const double* A, const double* B, double* C, int N, int size, int rank) {
    double start = omp_get_wtime();
    func(A, B, C, N, size, rank);
    return omp_get_wtime() - start;
}

int main(int argc, char** argv) {
    // Inicialização do MPI
    MPI_Init(&argc, &argv);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Número de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rank do processo atual

    ofstream csv;

    // Cabeçalho do CSV (apenas rank 0)
    if (rank == 0) {
        csv.open("resultados.csv", ios::app);
        if (!csv.is_open()) {
            cerr << "Erro ao abrir resultados.csv\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Cbeçalho do CSV

        if(csv.tellp() == 0){
            csv << "tamMatriz,tempoSequencial,"
                << "tempo2Thread,speedup2Thread,eficiencia2Thread,delta2Thread,"
                << "tempo4Thread,speedup4Thread,eficiencia4Thread,delta4Thread,"
                << "tempo8Thread,speedup8Thread,eficiencia8Thread,delta8Thread,"
                << "tempoMPI,speedupMPI,eficienciaMPI,deltaMPI\n";
        }
    }

    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    vector<int> num_threads = {2, 4, 8};

    // Percorre o vetor de tamanhos de matrizes
    for (int N : sizes) {
        double* A = nullptr;
        double* B = nullptr;
        double* C_seq = nullptr;
        double tempo_seq = 0.0;

        // Geração e cálculo sequencial (rank 0)
        if (rank == 0) {
            A = gerar_matriz(N);
            B = gerar_matriz(N);
            C_seq = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
            memset(C_seq, 0, N * N * sizeof(double));
            
            tempo_seq = medir_tempo_seq(dgemm_seq, A, B, C_seq, N);
        }

        // Broadcast do tempo sequencial para todos os processos
        MPI_Bcast(&tempo_seq, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // OpenMP (apenas rank 0)
        vector<double> tempos_omp(num_threads.size());
        vector<double> deltas_omp(num_threads.size());
        if (rank == 0) {
            for (size_t idx = 0 ; idx < num_threads.size() ; ++idx) {
                int nt = num_threads[idx];
                double* C_par = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
                memset(C_par, 0, N * N * sizeof(double));
                
                tempos_omp[idx] = medir_tempo_par(dgemm_par, A, B, C_par, N, nt);
                
                // Validação
                double delta;
                if (!validar_resultado(C_seq, C_par, N, delta)) {
                    cerr << "ERRO: Resultado OpenMP (" << nt << " threads) diverge! Delta=" << delta << "\n";
                }
                deltas_omp[idx] = delta;
                free(C_par);
            }
        }

        // Preparação para o MPI
        if (rank != 0) {
            A = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
            B = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
        }
        
        // Broadcast de A e B para todos os processos
        MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Alocação de C_mpi em todos os processos
        double* C_mpi = static_cast<double*>(aligned_alloc(64, N * N * sizeof(double)));
        memset(C_mpi, 0, N * N * sizeof(double));

        // Medição do tempo do MPI
        double tempo_mpi = medir_tempo_mpi(dgemm_mpi, A, B, C_mpi, N, size, rank);

        // Gravação dos resultados (rank 0)
        if (rank == 0) {
            // Validação MPI
            double delta_mpi;
            if (!validar_resultado(C_seq, C_mpi, N, delta_mpi)) {
                cerr << "ERRO: Resultado MPI diverge! Delta=" << delta_mpi << "\n";
            }

            csv << N << "," << fixed << setprecision(6) << tempo_seq;

            // Cálculo de métricas do OpenMP
            for (size_t idx = 0 ; idx < num_threads.size() ; ++idx) {
                double sp = tempo_seq / tempos_omp[idx];
                double ef = sp / num_threads[idx];
                csv << "," << tempos_omp[idx] << "," << sp << "," << ef << "," << scientific << deltas_omp[idx];
            }

            // Cálculo de métricas do MPI
            double sp_mpi = tempo_seq / tempo_mpi;
            double ef_mpi = sp_mpi / size;
            csv << "," << fixed << tempo_mpi << "," << sp_mpi << "," << ef_mpi << "," << scientific << delta_mpi << "\n";

            free(C_seq);
        }

        // Libera memória
        free(A);
        free(B);
        free(C_mpi);
    }

    // Fecha o arquivo CSV (rank 0)
    if (rank == 0) {
        csv.close();
    }

    // Finalização do MPI
    MPI_Finalize();
    return 0;
}