#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include "engine/tensor.h"

class Optimizer {
public:
    Optimizer(std::vector<Tensor*> params, float lr);
    virtual ~Optimizer();
    virtual void Step() = 0;
    virtual void ZeroGrad();

protected:
    std::vector<Tensor*> m_Parameters;
    float m_LearningRate;
};


class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr);
    virtual void Step() override;
};


class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    virtual void Step() override;

private:
    float m_Beta1;
    float m_Beta2;
    float m_Epsilon;
    int m_T;
    std::vector<Eigen::MatrixXf> m_M; // First moment
    std::vector<Eigen::MatrixXf> m_V; // Second moment
};


class AdamW : public Optimizer {
public:
    AdamW(std::vector<Tensor*> params, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weightDecay = 0.0);
    virtual void Step() override;

private:
    float m_Beta1;
    float m_Beta2;
    float m_Epsilon;
    int m_T;
    std::vector<Eigen::MatrixXf> m_M; // First moment
    std::vector<Eigen::MatrixXf> m_V; // Second moment
    float m_WeightDecay;
};


#endif // OPTIMIZERS_H