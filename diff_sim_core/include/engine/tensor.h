#ifndef CORE_H
#define CORE_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <memory>

class Tensor {
    friend class SGD;
    friend class Adam;
    friend class AdamW;
    friend Tensor relu(const Tensor& input);
    friend Tensor tanh(const Tensor& input);
    
public:
    // Constructor for arbitrary 2D shape (rows, cols)
    Tensor(); // Default constructor
    Tensor(int rows, int cols, bool requiresGrad = false);

    // Constructor for 1D column vector (size x 1)
    Tensor(int size, bool requiresGrad = false);

    // Constructor from 1D list (creates size x 1 column vector)
    Tensor(std::vector<float> dataList, bool requiresGrad = false);
    
    // Set value at specific row/col
    void Set(int r, int c, float value);

    // Get value
    float Get(int r, int c) const;

    // Backward function
    void Backward();

    // Set requires_grad
    void SetRequiresGrad(bool requiresGrad);

    // Zero gradient
    void ZeroGrad();

    // Dimensions
    int Rows() const;
    int Cols() const;

    // Accessors for Python bindings (Copy-based for now)
    Eigen::MatrixXf GetData() const;
    void SetData(const Eigen::MatrixXf& d);
    
    Eigen::MatrixXf GetGrad() const;
    void SetGrad(const Eigen::MatrixXf& g);

    bool GetRequiresGrad() const;

    // Pointer to underlying data (useful for binding to NumPy later)
    float* DataPtr();

    Tensor Sum();
    Tensor Sum(int axis); // Axis reduction
    Tensor Mean();
    Tensor Mean(int axis); // Axis reduction
    Tensor Max();
    Tensor Min();
    
    // Element-wise Math
    Tensor Exp();
    Tensor Log();
    Tensor Sqrt();
    Tensor Abs();
    Tensor Clamp(float minVal, float maxVal);

    // Trigonometry
    Tensor Sin();
    Tensor Cos();
    Tensor Pow(float exponent);

    // Core Ops
    Tensor Select(int idx) const; // Differentiable indexing
    static Tensor Stack(const std::vector<Tensor*>& tensors); // Differentiable stacking
    static Tensor Cat(const std::vector<Tensor*>& tensors, int dim); // Differentiable concatenation
    Tensor Reshape(int r, int c);

    // Operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(const Tensor& other) const;

    // Mathematical functions
    Tensor Transpose();
    Tensor Matmul(const Tensor& other);
    
    // Gaussian log probability for policy gradients
    static Tensor GaussianLogProb(const Tensor& action, const Tensor& mean, const Tensor& logStd);

private:
    // The backend: Dynamic size, Float type
    Eigen::MatrixXf m_Data;
    Eigen::MatrixXf m_Grad;
    bool m_bRequiresGrad = false;
    std::vector<Tensor*> m_Children;
    std::function<void(Tensor&)> m_BackwardFn;
};

#endif // CORE_H