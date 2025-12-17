#include "engine/optimizers.h"
#include <iostream>

Optimizer::Optimizer(std::vector<Tensor*> params, float lr)
    : m_Parameters(params), m_LearningRate(lr) {}

Optimizer::~Optimizer() {}

void Optimizer::ZeroGrad() {
    for (Tensor* pParam : m_Parameters) {
        pParam->ZeroGrad();
    }
}

SGD::SGD(std::vector<Tensor*> params, float lr)
    : Optimizer(params, lr) {}

void SGD::Step() {
    for (Tensor* pParam : m_Parameters) {
        if (!pParam->GetRequiresGrad()) continue;
        
        // Ensure gradient exists
        if (pParam->m_Grad.size() == 0) continue;

        // Basic SGD: p = p - lr * grad
        pParam->m_Data -= m_LearningRate * pParam->m_Grad;
    }
}

// ---------------- Adam ----------------

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2, float epsilon)
    : Optimizer(params, lr), m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), m_T(0) {
    
    // Initialize m and v with zeros matching param shapes
    for (Tensor* pParam : params) {
        m_M.push_back(Eigen::MatrixXf::Zero(pParam->Rows(), pParam->Cols()));
        m_V.push_back(Eigen::MatrixXf::Zero(pParam->Rows(), pParam->Cols()));
    }
}

void Adam::Step() {
    m_T++;
    for (size_t i = 0; i < m_Parameters.size(); ++i) {
        Tensor* pParam = m_Parameters[i];
        if (!pParam->GetRequiresGrad() || pParam->m_Grad.size() == 0) continue;

        // Current Gradient
        const Eigen::MatrixXf& g = pParam->m_Grad;

        // Update biased first moment estimate
        m_M[i] = m_Beta1 * m_M[i] + (1.0f - m_Beta1) * g;

        // Update biased second raw moment estimate
        m_V[i] = m_Beta2 * m_V[i] + (1.0f - m_Beta2) * g.array().square().matrix();

        // Compute bias-corrected first moment estimate
        Eigen::MatrixXf mHat = m_M[i] / (1.0f - std::pow(m_Beta1, m_T));

        // Compute bias-corrected second raw moment estimate
        Eigen::MatrixXf vHat = m_V[i] / (1.0f - std::pow(m_Beta2, m_T));

        // Update parameters
        // theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        pParam->m_Data.array() -= m_LearningRate * mHat.array() / (vHat.array().sqrt() + m_Epsilon);
    }
}


// ---------------- AdamW ----------------


AdamW::AdamW(std::vector<Tensor*> params, float lr, float beta1, float beta2, float epsilon, float weightDecay)
    : Optimizer(params, lr), m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), m_WeightDecay(weightDecay), m_T(0) {
    
    for (Tensor* pParam : params) {
        m_M.push_back(Eigen::MatrixXf::Zero(pParam->Rows(), pParam->Cols()));
        m_V.push_back(Eigen::MatrixXf::Zero(pParam->Rows(), pParam->Cols()));
    }
}

void AdamW::Step() {
    m_T++;
    for (size_t i = 0; i < m_Parameters.size(); ++i) {
        Tensor* pParam = m_Parameters[i];
        if (!pParam->GetRequiresGrad() || pParam->m_Grad.size() == 0) continue;
        
        // AdamW Decoupled Weight Decay
        if (m_WeightDecay > 0) {
            pParam->m_Data -= m_LearningRate * m_WeightDecay * pParam->m_Data;
        }
        
        const Eigen::MatrixXf& g = pParam->m_Grad;
        m_M[i] = m_Beta1 * m_M[i] + (1.0f - m_Beta1) * g;
        m_V[i] = m_Beta2 * m_V[i] + (1.0f - m_Beta2) * g.array().square().matrix();
        Eigen::MatrixXf mHat = m_M[i] / (1.0f - std::pow(m_Beta1, m_T));
        Eigen::MatrixXf vHat = m_V[i] / (1.0f - std::pow(m_Beta2, m_T));
        pParam->m_Data.array() -= m_LearningRate * mHat.array() / (vHat.array().sqrt() + m_Epsilon);
    }
}
