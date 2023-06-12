#include "Header.h"
#include <sys/time.h>
#include <cstdio>
#include <unistd.h>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define u32 unsigned int
#define u64 unsigned long long

template<typename T>
class Evaluater
{
public:
    std::vector<T*> evs;
    u32 num;
    Evaluater(u32 _num) {
        num = _num;
        for(u32 i = 0; i < num; i ++) {
            T *f = new T();
            evs.push_back(f);
        }
    }
    void evalAll(double **Xs, double *ans) {
        #pragma omp parallel for // 此处使用OMP
        for(u32 i = 0; i < num; i ++) {
            ans[i] = evs[i] -> compute(Xs[i]);
        }
    }
    double evalOne(double *X) {
        return evs[0] -> compute(X);
    }
};

double random1() {
    return double(rand()) / RAND_MAX;
}

double random_between(double low, double high) {
    return low + random1() * (high - low);
}

const u32 GA_CREATURE_NUM = 500;
const u32 GA_ITER_NUM = 5000;
const u32 GA_PRINT_INTERVAL = 200;
const double GA_INIT_LOW = -10;
const double GA_INIT_HIGH = 10;
const double GA_CROSSOVER_PROB = 0.8;
const double GA_CROSSOVER_SWAP_PROB = 0.4;
const double GA_MUTATION_PROB = 0.1;
const double GA_MUTATION_PART = 0.005;
const double GA_MUTATION_LOW = -2;
const double GA_MUTATION_HIGH = 2;

template<typename T>
class GA
{
private:
    std::shared_ptr<Evaluater<T>> ev;
    u32 xdim;
    double *creatures[GA_CREATURE_NUM], *best = nullptr, results[GA_CREATURE_NUM], time_passed = 0.0;
public:
    GA(
        std::shared_ptr<Evaluater<T>> _ev, 
        u32 _xdim
    ): ev(_ev), xdim{_xdim} {};
    ~GA() {
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            delete[] creatures[i];
        }
        if(best) {
            delete[] best;
        }
    }
    double random() {
        return (double)rand() / RAND_MAX;
    }
    u32 get_result_argmax() {
        double result = results[0];
        u32 maxi = 0;
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            if(results[i] > result) {
                result = results[i];
                maxi = i;
            }
        }
        return maxi;
    }
    u32 get_result_argmin() {
        double result = results[0];
        u32 mini = 0;
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            if(results[i] < result) {
                result = results[i];
                mini = i;
            }
        }
        return mini;
    }
    void copy(double *from, double *to){
        memcpy(to, from, xdim * sizeof(double));
    }
    double result_of(double *x) {
        return ev -> evalOne(x);
    }
    void keep_best(bool compare) {
        u32 max_i = get_result_argmax();
        if(compare && result_of(best) < results[max_i]) {
            return;
        }
        copy(creatures[max_i], best);
    }
    double get_relative_score(double cur, double max, double min) {
        return log((max - cur) / (max - min) + 1.145141919810);
    }
    double get_relative_score_sum(double max, double min) {
        double ans = 0.0;
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            ans += get_relative_score(results[i], max, min);
        }
        return ans;
    }
    void init() {
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            creatures[i] = new double[xdim];
            for(u32 j = 0; j < xdim; j ++) {
                creatures[i][j] = random() * (GA_INIT_HIGH - GA_INIT_LOW) + GA_INIT_LOW;
            }
        }
        best = new double[xdim];
        copy(creatures[0], best);
    }
    void evaluate() {

        // auto start = std::chrono::high_resolution_clock::now();
        ev -> evalAll(creatures, results);
        // for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
        //     results[i] = result_of(creatures[i]);
        // }

        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // time_passed += duration.count();

    }
    void select() {

        double *new_creatures[GA_CREATURE_NUM];
        double relative_score[GA_CREATURE_NUM];
        double min_result = result_of(creatures[get_result_argmin()]);
        double max_result = result_of(creatures[get_result_argmax()]);
        double score_sum = get_relative_score_sum(max_result, min_result);

        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            relative_score[i] = get_relative_score(results[i], max_result, min_result);
        }

        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            new_creatures[i] = new double[xdim];
            double p = random();
            u32 j;
            for(j = 0; j < GA_CREATURE_NUM; j ++) {
                p -= relative_score[j] / score_sum;
                if(p <= 0) {
                    // cout << j << " " 
                    //     << get_relative_score(results[j], max_score, min_score) << " " 
                    //     << score_sum << " " 
                    //     << get_relative_score(min_score, max_score, min_score) << endl;
                    copy(creatures[j], new_creatures[i]);
                    break;
                }
            }
            if(j >= GA_CREATURE_NUM) {
                copy(best, new_creatures[i]);
            }
        }
        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            copy(new_creatures[i], creatures[i]);
            delete[] new_creatures[i];

            // creatures[i] = new_creatures[i];
            // delete[] creatures[i];
        }


    }
    void crossover() {


        for(u32 i = 0; i+1 < GA_CREATURE_NUM; i += 2) {
            if(random() > GA_CROSSOVER_PROB) {
                continue;
            }
            for(u32 j = 0; j < xdim; j ++) {
                if(random() > GA_CROSSOVER_SWAP_PROB) {
                    continue;
                }
                swap(creatures[i][j], creatures[i+1][j]);
            }
        }

    }
    void mutate() {

        for(u32 i = 0; i < GA_CREATURE_NUM; i ++) {
            if(random() > GA_MUTATION_PROB) {
                continue;
            }
            for(u32 j = 0; j < xdim; j ++) {
                if(random() > GA_MUTATION_PART) {
                    continue;
                }
                double diff = GA_MUTATION_LOW + random() * (GA_MUTATION_HIGH - GA_MUTATION_LOW);
                creatures[i][j] += diff;
            }
        }
    }
    void print_status() {
        cout << "best: [ ..., ";
        for (u32 i = xdim - 5; i < xdim; i++)
        {
            cout << best[i] << ", ";
        }
        double result = ev -> evalOne(best);
        cout << "] -> result: " << result
            << ", log(result): " << log(result) << endl;
        // cout << "time_passed: " << time_passed << "us" << endl;
    }
    void run(double *ans) {
        init();
        evaluate();
        keep_best(false);
        for(u32 i = 0; i < GA_ITER_NUM; i ++) {
            if (i % GA_PRINT_INTERVAL == 0) {
                cout << i << ": ---" << endl;
            }
            select();
            crossover();
            mutate();
            evaluate();
            keep_best(true);
            if(i % GA_PRINT_INTERVAL == 0) {
                print_status();
            }
        }
        copy(best, ans);
    }
};

const u32 PSO_ITER_NUM = 5000;
const u32 PSO_PARTICAL_NUM = 50; 
const u32 PSO_PRINT_INTERVAL = 500;
const double PSO_POSITION_LOW = -100;
const double PSO_POSITION_HIGH = 100;
const double PSO_CONST_W = 0.9; // 惯性因子
const double PSO_CONST_C1 = 1.5; // 学习因子1
const double PSO_CONST_C2 = 1.5; // 学习因子2
const double PSO_VMAX = 3;

template<typename T>
class PSO
{
private:
    double *ps[PSO_PARTICAL_NUM], *vs[PSO_PARTICAL_NUM], 
            *best_ps[PSO_PARTICAL_NUM], *gbest_p;
    double results[PSO_PARTICAL_NUM], best_results[PSO_PARTICAL_NUM], gbest_result;
    std::shared_ptr<Evaluater<T>> ev;
    u32 xdim;
public:

    PSO(
        std::shared_ptr<Evaluater<T>> _ev, 
        u32 _xdim
    ): ev(_ev), xdim{_xdim} {
    }
    double random() {
        return (double)rand() / RAND_MAX;
    }
    void copy(double *from, double *to){
        memcpy(to, from, xdim * sizeof(double));
    }
    void eval_bestps() {
        ev -> evalAll(best_ps, best_results);
        gbest_result = ev -> evalOne(gbest_p);
    }
    double fitness(double result) {
        return abs(result);
    }   
    void init() {
        for(u32 i = 0; i < PSO_PARTICAL_NUM; i ++) {
            ps[i] = new double[xdim];
            vs[i] = new double[xdim];
            best_ps[i] = new double[xdim];
            for(u32 j = 0; j < xdim; j ++) {
                best_ps[i][j] = ps[i][j] = random();
                vs[i][j] = 0;
            }
        }
        gbest_p = new double[xdim];
        copy(best_ps[0], gbest_p);
    }
    void evaluate() {
        ev -> evalAll(ps, results);
    }
    void save_best() {
        for(u32 i = 0; i < PSO_PARTICAL_NUM; i ++) {
            if(fitness(results[i]) < fitness(best_results[i])) {
                copy(ps[i], best_ps[i]);
                best_results[i] = results[i];
            }
            if(fitness(best_results[i]) < fitness(gbest_result)) {
                copy(best_ps[i], gbest_p);
                gbest_result = best_results[i];
            }
        }
    }
    void update_pos() {
        for(u32 i = 0; i < PSO_PARTICAL_NUM; i ++) {
            #pragma omp parallel for
            for(u32 j = 0; j < xdim; j ++) {
                ps[i][j] += vs[i][j];
                if(ps[i][j] < PSO_POSITION_LOW) {
                    ps[i][j] = PSO_POSITION_LOW * (0.99 + random() * 0.01);
                }
                if(ps[i][j] > PSO_POSITION_HIGH) {
                    ps[i][j] = PSO_POSITION_HIGH* (0.99 + random() * 0.01);
                }
            }
        }
    }
    void update_vs() {
        double r1 = random(), r2 = random();

        for(u32 i = 0; i < PSO_PARTICAL_NUM; i ++) {
            #pragma omp parallel for
            for(u32 j = 0; j < xdim; j ++) {
                vs[i][j] = PSO_CONST_W * vs[i][j] + \
                    PSO_CONST_C1 * r1 * (best_ps[i][j] - ps[i][j]) + \
                    PSO_CONST_C2 * r2 * (gbest_p[j] - ps[i][j]);
                if(abs(vs[i][j]) > PSO_VMAX) { // 这个提升很大
                    vs[i][j] *= PSO_VMAX / abs(vs[i][j]);
                }
            }
        }
    }
    void run(double *ans) {
        init();
        eval_bestps();
        for(u32 i = 0; i < PSO_ITER_NUM; i ++) {
            update_pos();
            evaluate();
            save_best();
            update_vs();
            if(i % PSO_PRINT_INTERVAL == 0) {
                cout << "gbest_result=" << ev -> evalOne(gbest_p) 
                    << ", gbest_p[-1]=" << gbest_p[xdim - 1] << endl;
                cout << "ps[0][-1]=" << ps[0][xdim-1] 
                    << ", vs[0][-1]=" << vs[0][xdim-1] << endl;
            }
        }
        copy(gbest_p, ans);
    }
};

const u32 DE_ITER_NUM = 10000;
const u32 DE_CREATURES_NUM = 100;
const double DE_BOUND_LOW = -10;
const double DE_BOUND_HIGH = 10;
const double DE_MUTATION_RATE = 0.5;
const double DE_CROSSOVER_RATE = 0.1;

template<typename T>
class DE 
{
private:
    std::shared_ptr<Evaluater<T>> ev;
    u32 xdim;
    double *creatures[DE_CREATURES_NUM], results[DE_CREATURES_NUM];
public:
    DE(
        std::shared_ptr<Evaluater<T>> _ev, 
        u32 _xdim
    ): ev(_ev), xdim{_xdim} {
    }
    void sample3(u32 &a, u32 &b, u32 &c) {
        a = random1() * double(DE_CREATURES_NUM);
        b = random1() * double(DE_CREATURES_NUM);
        c = random1() * double(DE_CREATURES_NUM);
        if(a == b || b == c || c == a) {
            sample3(a, b, c);
        }
    }
    void copy(double *from, double *to){
        memcpy(to, from, xdim * sizeof(double));
    }
    double fitness(double result) {
        return abs(result);
    }
    void init() {
        for(u32 i = 0; i < DE_CREATURES_NUM; i ++) {
            creatures[i] = new double[xdim];
            for(u32 j = 0; j < xdim; j ++) {
                creatures[i][j] = random_between(DE_BOUND_LOW, DE_BOUND_HIGH);
            }
        }
        ev -> evalAll(creatures, results);
    }
    void get_candidates(double *candidates[DE_CREATURES_NUM]) {
        #pragma omp parallel for
        for(u32 i = 0; i < DE_CREATURES_NUM; i ++) {
            u32 a, b, c, d;
            sample3(a, b, c);
            d = random_between(0, xdim);
            for(u32 j = 0; j < xdim; j ++) {
                if(random1() < DE_CROSSOVER_RATE || j == d) {
                    candidates[i][j] = creatures[a][j] + \
                        DE_MUTATION_RATE * (
                            creatures[b][j] - creatures[c][j]
                        );
                }else{
                    candidates[i][j] = creatures[i][j];
                }

            }
        }
    }
    void update_candidates() {
        double *candidates[DE_CREATURES_NUM], candidates_results[DE_CREATURES_NUM];
        for(u32 i = 0; i < DE_CREATURES_NUM; i ++) {
            candidates[i] = new double[xdim];
        }
        
        get_candidates(candidates);

        ev -> evalAll(candidates, candidates_results);

        for(u32 i = 0; i < DE_CREATURES_NUM; i ++) {
            if(fitness(candidates_results[i]) < fitness(results[i])) {
                copy(candidates[i], creatures[i]);
                results[i] = candidates_results[i];
            }
            delete[] candidates[i];
        }

    }
    void run(double *ans) {
        init();
        double best_result;
        for(u32 i = 0; i < DE_ITER_NUM; i ++) {
            update_candidates();
            if (i % 200 == 1) {
                cout << "i=" << i 
                    << ", creatures[0][-1]=" << creatures[0][xdim-1] 
                    << ", results[0]=" << results[0]
                    << ", log(results[0])=" << log(results[0])
                    << endl;
            }
        }
        best_result = results[0];
        copy(creatures[0], ans);
        for(u32 i = 1; i < DE_CREATURES_NUM; i ++) {
            if(fitness(results[i]) < fitness(best_result)) {
                copy(creatures[i], ans);
                best_result = results[i];
            }
        }
        
    }
};


int main()
{
    srand(static_cast<unsigned int>(time(nullptr)));

    double *X;
    u32 dim = 1000;
    
    // std::shared_ptr<Evaluater< F15 >> ev = std::make_shared<Evaluater< F15 >>(GA_CREATURE_NUM);
    // std::shared_ptr<GA< F15 >> op = std::make_shared<GA< F15 >>(ev, dim);

    // std::shared_ptr<Evaluater< F15 >> ev = std::make_shared<Evaluater< F15 >>(PSO_PARTICAL_NUM);
    // std::shared_ptr<PSO< F15 >> op = std::make_shared<PSO< F15 >>(ev, dim);

    std::shared_ptr<Evaluater< F15 >> ev = std::make_shared<Evaluater< F15 >>(DE_CREATURES_NUM);
    std::shared_ptr<DE< F15 >> op = std::make_shared<DE< F15 >>(ev, dim);

    X = new double[dim];
    op -> run(X);
    cout << X << " -> ";
    for (u32 i = dim - 10; i < dim; i++)
    {
        cout << X[i] << " ";
    }
    cout << endl;
    cout << " F15 (X): " << ev -> evalOne(X) << endl;                                      

    // X = new double[dim];
    // double ans_sum = 0;
    // u32 times = 0;
    // for(times = 0; times < 5; times ++) {
    //     op -> run(X);
    //     ans_sum += ev -> evalOne(X);
    //     cout << "ans=" << ans_sum / (times + 1) << endl;
    // }
    // cout << " F15 (X): " << ans_sum / times << endl;

    std::ofstream output_file("output-DE-F15.txt");

    if (output_file.is_open()) { 
        for (u32 i = 0; i < dim; i++)
        {
            output_file << X[i] << endl;
        }
        output_file.close();
    } else { // 文件打开失败
        std::cout << "Unable to open file for writing.\n";
    }


    for (u32 i = 0; i < dim; i++)
    {
        X[i] = 0;
    }

    cout << " F15 (0): " << ev -> evalOne(X) << endl;

    return 0;
}