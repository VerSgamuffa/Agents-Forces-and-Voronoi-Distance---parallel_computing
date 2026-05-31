#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>

struct Vector2D {
    double x = 0.0;
    double y = 0.0;
    Vector2D operator+(const Vector2D& o) const { return {x + o.x, y + o.y}; }
    Vector2D operator-(const Vector2D& o) const { return {x - o.x, y - o.y}; }
    Vector2D operator*(double s) const { return {x * s, y * s}; }
    Vector2D& operator+=(const Vector2D& o) { x += o.x; y += o.y; return *this; }
    double norm() const { return std::sqrt(x * x + y * y); }
};

double dot_product(const Vector2D& v1, const Vector2D& v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

// taglio poligono
std::vector<Vector2D> clip_polygon(const std::vector<Vector2D>& poly, const Vector2D& M, const Vector2D& N) {
    std::vector<Vector2D> clipped;
    if (poly.empty()) return clipped;

    for (size_t i = 0; i < poly.size(); ++i) {
        Vector2D A = poly[i];
        Vector2D B = poly[(i + 1) % poly.size()];

        double dist_A = dot_product(A - M, N);
        double dist_B = dot_product(B - M, N);

        bool A_inside = dist_A <= 1e-9;
        bool B_inside = dist_B <= 1e-9;

        if (A_inside) clipped.push_back(A);
        if (A_inside != B_inside) {
            double t = dist_A / (dist_A - dist_B);
            Vector2D intersect = {A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)};
            clipped.push_back(intersect);
        }
    }
    return clipped;
}

// centroide poligono
Vector2D polygon_centroid(const std::vector<Vector2D>& poly, const Vector2D& fallback) {
    if (poly.empty()) return fallback;
    double area = 0.0, cx = 0.0, cy = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        Vector2D P0 = poly[i];
        Vector2D P1 = poly[(i + 1) % poly.size()];
        double cross = (P0.x * P1.y - P1.x * P0.y);
        area += cross;
        cx += (P0.x + P1.x) * cross;
        cy += (P0.y + P1.y) * cross;
    }
    area *= 0.5;
    if (std::abs(area) < 1e-9) return fallback;
    return {cx / (6.0 * area), cy / (6.0 * area)};
}

// fase 1 sequenziale
std::vector<Vector2D> compute_exact_voronoi_centroids_seq(const std::vector<Vector2D>& points, double W, double H) {
    int n = points.size();
    std::vector<Vector2D> centroids(n);
    std::vector<Vector2D> bbox = {{0.0, 0.0}, {W, 0.0}, {W, H}, {0.0, H}};

    for (int i = 0; i < n; ++i) {
        std::vector<Vector2D> cell = bbox;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            Vector2D M = {(points[i].x + points[j].x) * 0.5, (points[i].y + points[j].y) * 0.5};
            Vector2D N = {points[j].x - points[i].x, points[j].y - points[i].y};
            cell = clip_polygon(cell, M, N);
        }
        centroids[i] = polygon_centroid(cell, points[i]);
    }
    return centroids;
}

// fase 1 parallela
std::vector<Vector2D> compute_exact_voronoi_centroids_omp(const std::vector<Vector2D>& points, double W, double H) {
    int n = points.size();
    std::vector<Vector2D> centroids(n);
    std::vector<Vector2D> bbox = {{0.0, 0.0}, {W, 0.0}, {W, H}, {0.0, H}};

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        std::vector<Vector2D> cell = bbox;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            Vector2D M = {(points[i].x + points[j].x) * 0.5, (points[i].y + points[j].y) * 0.5};
            Vector2D N = {points[j].x - points[i].x, points[j].y - points[i].y};
            cell = clip_polygon(cell, M, N);
        }
        centroids[i] = polygon_centroid(cell, points[i]);
    }
    return centroids;
}

// connessione grafo
bool is_connected(const std::vector<std::vector<int>>& A, int n_nodes) {
    std::vector<bool> visited(n_nodes, false);
    std::vector<int> queue;
    queue.push_back(0);
    visited[0] = true;
    size_t head = 0;
    while (head < queue.size()) {
        int curr = queue[head++];
        for (int v = 0; v < n_nodes; ++v) {
            if (A[curr][v] == 1 && !visited[v]) {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }
    for (bool v : visited) { if (!v) return false; }
    return true;
}

std::vector<std::vector<int>> generate_connected_graph(int n_nodes, double p, std::mt19937& gen) {
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    while (true) {
        std::vector<std::vector<int>> A(n_nodes, std::vector<int>(n_nodes, 0));
        for (int i = 0; i < n_nodes; ++i) {
            for (int j = i + 1; j < n_nodes; ++j) {
                if (dis(gen) < p) { A[i][j] = 1; A[j][i] = 1; }
            }
        }
        if (is_connected(A, n_nodes)) return A;
    }
}

int main() {
    const int n_agents = 100;
    //const double W = 10.0, H = 10.0;
    const double W = 20.0, H = 20.0;
    const double step = 0.10;
    const int frames = 300;
    const double rho0 = 5.0, b = 2.0, rho = 7.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_agent(-0.5, 0.5);

    Vector2D center = {W / 2.0, H / 2.0};
    double spread = 5.0;

    // generazione posizioni iniziali fisse
    std::vector<Vector2D> initial_agents(n_agents);
    for (int i = 0; i < n_agents; ++i) {
        initial_agents[i].x = center.x + dis_agent(gen) * spread;
        initial_agents[i].y = center.y + dis_agent(gen) * spread;
    }

    std::vector<std::vector<int>> A = generate_connected_graph(n_agents, 0.03, gen);

    // salvataggio adiacenza
    std::ofstream out_adj("adiacenza.csv");
    for(int i = 0; i < n_agents; ++i){
        for(int j = 0; j < n_agents; ++j){
            out_adj << A[i][j] << (j == n_agents-1 ? "" : ",");
        }
        out_adj << "\n";
    }
    out_adj.close();


    std::cout << "======================================================\n";
    std::cout << "     1. AVVIO RUN SEQUENZIALE            \n";
    std::cout << "======================================================\n";

    std::vector<Vector2D> agents_seq = initial_agents;
    double seq_voronoi = 0.0, seq_proposed = 0.0, seq_forces = 0.0, seq_io = 0.0;

    std::ofstream csv_seq("traiettorie_sequenziali.csv");
    csv_seq << "frame,agent_id,x,y\n";

    auto seq_total_start = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < frames; ++frame) {
        // voronoi sequenziale
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Vector2D> centroids = compute_exact_voronoi_centroids_seq(agents_seq, W, H);
        auto end = std::chrono::high_resolution_clock::now();
        seq_voronoi += std::chrono::duration<double, std::milli>(end - start).count();

        // posizioni proposte
        start = std::chrono::high_resolution_clock::now();
        std::vector<Vector2D> to_centroid(n_agents);
        std::vector<Vector2D> proposed_positions(n_agents);
        for (int i = 0; i < n_agents; ++i) {
            to_centroid[i] = centroids[i] - agents_seq[i];
            proposed_positions[i] = agents_seq[i] + to_centroid[i] * step;
        }
        end = std::chrono::high_resolution_clock::now();
        seq_proposed += std::chrono::duration<double, std::milli>(end - start).count();

        // forze attrattive sequenziali
        start = std::chrono::high_resolution_clock::now();
        std::vector<Vector2D> force_attr(n_agents, {0.0, 0.0});
        for (int i = 0; i < n_agents; ++i) {
            Vector2D total_local_force = {0.0, 0.0};
            for (int j = 0; j < n_agents; ++j) {
                if (i == j || A[i][j] != 1) continue;
                Vector2D direction = proposed_positions[j] - proposed_positions[i];
                double dist = direction.norm();
                if (dist > rho0) {
                    double x_val = (1.0 / (dist - rho)) - (1.0 / (rho0 - rho));
                    double force_strength = -std::pow(x_val, b - 1) / std::pow(rho - dist, 2);
                    total_local_force += direction * (force_strength / dist);
                }
            }
            force_attr[i] = total_local_force;
        }
        end = std::chrono::high_resolution_clock::now();
        seq_forces += std::chrono::duration<double, std::milli>(end - start).count();

        // scrittura e update
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_agents; ++i) {
            agents_seq[i] += to_centroid[i] * step + force_attr[i];
            csv_seq << frame << "," << i << "," << agents_seq[i].x << "," << agents_seq[i].y << "\n";
        }
        end = std::chrono::high_resolution_clock::now();
        seq_io += std::chrono::duration<double, std::milli>(end - start).count();
    }
    auto seq_total_end = std::chrono::high_resolution_clock::now();
    double seq_total_time = std::chrono::duration<double, std::milli>(seq_total_end - seq_total_start).count();
    csv_seq.close();

    // salvataggio tempi sequenziali
    std::ofstream bench_seq_file("benchmark_sequenziale.csv");
    bench_seq_file << "total_time_ms,voronoi_ms,proposed_ms,forces_ms,io_ms\n";
    bench_seq_file << seq_total_time << "," << seq_voronoi << "," << seq_proposed << "," << seq_forces << "," << seq_io << "\n";
    bench_seq_file.close();

    // stampa
    std::cout << "\n------------------------------------------------------\n";
    std::cout << "VERSIONE SEQUENZIALE PURA | Tempo Totale: " << seq_total_time / 1000.0 << " s\n";
    std::cout << "1. Voronoi Esatto (SEQ):      " << std::fixed << std::setprecision(4) << seq_voronoi / 1000.0  << " s\t| " << (seq_voronoi/seq_total_time)*100 << "%\n";
    std::cout << "2. Posizioni Proposte:         " << seq_proposed / 1000.0 << " s\t| " << (seq_proposed/seq_total_time)*100 << "%\n";
    std::cout << "3. Calcolo Forze (SEQ):        " << seq_forces / 1000.0  << " s\t| " << (seq_forces/seq_total_time)*100 << "%\n";
    std::cout << "4. Scrittura CSV & Update I/O: " << seq_io / 1000.0  << " s\t| " << (seq_io/seq_total_time)*100 << "%\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "\n-> Run Sequenziale Completata. File 'traiettorie_sequenziali.csv' e 'benchmark_sequenziale.csv' salvati.\n";


    std::cout << "\n======================================================\n";
    std::cout << "     2. AVVIO RUN PARALLELA CON OPENMP (1 -> 32 THREADS) \n";
    std::cout << "======================================================\n";

    std::ofstream bench_par_file("benchmark_parallelo.csv");
    bench_par_file << "threads,total_time_ms,voronoi_ms,proposed_ms,forces_ms,io_ms\n";

    for (int num_threads = 1; num_threads <= 32; ++num_threads) {

        omp_set_num_threads(num_threads);
        std::vector<Vector2D> agents_omp = initial_agents;

        double duration_voronoi = 0.0;
        double duration_proposed = 0.0;
        double duration_forces = 0.0;
        double duration_io_update = 0.0;

        std::string filename_traiettorie = "traiettorie_parallele_th_" + std::to_string(num_threads) + ".csv";
        std::ofstream csv_omp(filename_traiettorie);
        csv_omp << "frame,agent_id,x,y\n";

        auto total_loop_start = std::chrono::high_resolution_clock::now();

        for (int frame = 0; frame < frames; ++frame) {

            // calcolo voronoi (OpenMP)
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Vector2D> centroids = compute_exact_voronoi_centroids_omp(agents_omp, W, H);
            auto end = std::chrono::high_resolution_clock::now();
            duration_voronoi += std::chrono::duration<double, std::milli>(end - start).count();

            // posizioni proposte
            start = std::chrono::high_resolution_clock::now();
            std::vector<Vector2D> to_centroid(n_agents);
            std::vector<Vector2D> proposed_positions(n_agents);
            for (int i = 0; i < n_agents; ++i) {
                to_centroid[i] = centroids[i] - agents_omp[i];
                proposed_positions[i] = agents_omp[i] + to_centroid[i] * step;
            }
            end = std::chrono::high_resolution_clock::now();
            duration_proposed += std::chrono::duration<double, std::milli>(end - start).count();

            // calcolo forze attrattive (OpenMP)
            start = std::chrono::high_resolution_clock::now();
            std::vector<Vector2D> force_attr(n_agents, {0.0, 0.0});

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_agents; ++i) {
                Vector2D total_local_force = {0.0, 0.0};
                for (int j = 0; j < n_agents; ++j) {
                    if (i == j || A[i][j] != 1) continue;
                    Vector2D direction = proposed_positions[j] - proposed_positions[i];
                    double dist = direction.norm();
                    if (dist > rho0) {
                        double x_val = (1.0 / (dist - rho)) - (1.0 / (rho0 - rho));
                        double force_strength = -std::pow(x_val, b - 1) / std::pow(rho - dist, 2);
                        total_local_force += direction * (force_strength / dist);
                    }
                }
                force_attr[i] = total_local_force;
            }
            end = std::chrono::high_resolution_clock::now();
            duration_forces += std::chrono::duration<double, std::milli>(end - start).count();

            // aggiornamento e scrittura
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_agents; ++i) {
                agents_omp[i] += to_centroid[i] * step + force_attr[i];
                csv_omp << frame << "," << i << "," << agents_omp[i].x << "," << agents_omp[i].y << "\n";
            }
            end = std::chrono::high_resolution_clock::now();
            duration_io_update += std::chrono::duration<double, std::milli>(end - start).count();
        }

        auto total_loop_end = std::chrono::high_resolution_clock::now();
        double total_loop_time = std::chrono::duration<double, std::milli>(total_loop_end - total_loop_start).count();
        csv_omp.close();

        // scrittura dei log dei tempi sul CSV di benchmark parallelo
        bench_par_file << num_threads << ","
                       << total_loop_time << ","
                       << duration_voronoi << ","
                       << duration_proposed << ","
                       << duration_forces << ","
                       << duration_io_update << "\n";

        // stampa a schermo per ogni thread
        std::cout << "\n------------------------------------------------------\n";
        std::cout << "Thread OpenMP: " << num_threads << " | Tempo Totale: " << total_loop_time / 1000.0 << " s\n";
        std::cout << "1. Voronoi Esatto (OMP):      " << std::fixed << std::setprecision(4) << duration_voronoi / 1000.0  << " s\t| " << (duration_voronoi/total_loop_time)*100 << "%\n";
        std::cout << "2. Posizioni Proposte:         " << duration_proposed / 1000.0 << " s\t| " << (duration_proposed/total_loop_time)*100 << "%\n";
        std::cout << "3. Calcolo Forze (OMP):        " << duration_forces / 1000.0  << " s\t| " << (3/total_loop_time)*100 << "%\n";
        std::cout << "4. Scrittura CSV & Update I/O: " << duration_io_update / 1000.0  << " s\t| " << (duration_io_update/total_loop_time)*100 << "%\n";
    }

    bench_par_file.close();
    std::cout << "\n======================================================\n";
    std::cout << " BENCHMARK COMPLETATO!\n";
    std::cout << " - File Tempi Sequenziali:  'benchmark_sequenziale.csv'\n";
    std::cout << " - File Tempi Paralleli:    'benchmark_parallelo.csv'\n";
    std::cout << " - File Traiettorie:        'traiettorie_sequenziali.csv' e 'traiettorie_parallele_th_X.csv'\n";
    std::cout << "======================================================\n";

    return 0;
}