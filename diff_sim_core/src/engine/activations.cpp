#include "engine/activations.h"
#include <iostream>

Tensor relu(const Tensor& input) {
    Tensor result(input.GetData().rows(), input.GetData().cols(), false);
    result.SetData(input.GetData().cwiseMax(0.0f));

    if (input.GetRequiresGrad()) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        
        // We need to store 'input' to compute mask during backward. Be careful with const reference.
        Tensor* pInput = (Tensor*)&input;
        result.m_Children.push_back(pInput);

        result.m_BackwardFn = [pInput](Tensor& self) {
            if (pInput->GetRequiresGrad()) {
                Eigen::MatrixXf mask = (pInput->GetData().array() > 0.0f).cast<float>();
                pInput->m_Grad.array() += mask.array() * self.m_Grad.array();
            }
        };
    }
    return result;
}

Tensor tanh(const Tensor& input) {
    Tensor result(input.GetData().rows(), input.GetData().cols(), false);
    result.SetData(input.GetData().array().tanh());

    if (input.GetRequiresGrad()) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        
        Tensor* pInput = (Tensor*)&input;
        result.m_Children.push_back(pInput);

        // Proper closure with input capture
        result.m_BackwardFn = [pInput](Tensor& self) {
             if (pInput->GetRequiresGrad()) {
                 // dy/dx = 1 - y^2
                 // y is self.m_Data
                 Eigen::MatrixXf deriv = 1.0f - self.GetData().array().square();
                 pInput->m_Grad.array() += deriv.array() * self.m_Grad.array();
             }
        };
    }
    return result;
}
