#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include "engine/tensor.h"

class Optimizer {
public:
    Optimizer(std::vector<Tensor*> params, float lr);
    virtual ~Optimizer();
    virtual void step() = 0;
    virtual void zero_grad();

protected:
    std::vector<Tensor*> parameters;
    float learning_rate;
};


class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr);
    virtual void step() override;
};


class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    virtual void step() override;

private:
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<Eigen::MatrixXf> m; // First moment
    std::vector<Eigen::MatrixXf> v; // Second moment
};


class AdamW : public Optimizer {
public:
    AdamW(std::vector<Tensor*> params, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay = 0.0);
    virtual void step() override;

private:
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<Eigen::MatrixXf> m; // First moment
    std::vector<Eigen::MatrixXf> v; // Second moment
    float weight_decay;
};


#endif // OPTIMIZERS_H