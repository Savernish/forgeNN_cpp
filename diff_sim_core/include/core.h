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
public:
    // Constructor for arbitrary 2D shape (rows, cols)
    Tensor(int rows, int cols, bool requires_grad = false);

    // Constructor for 1D column vector (size x 1)
    Tensor(int size, bool requires_grad = false);

    // Constructor from 1D list (creates size x 1 column vector)
    Tensor(std::vector<float> data_list, bool requires_grad = false);
    
    // Set value at specific row/col
    void set(int r, int c, float value);

    // Get value
    float get(int r, int c) const;

    // Backward function
    void backward();

    // Set requires_grad
    void set_requires_grad(bool requires_grad);

    // Zero gradient
    void zero_grad();

    // Dimensions
    int rows() const;
    int cols() const;

    // Accessors for Python bindings (Copy-based for now)
    Eigen::MatrixXf get_data() const;
    void set_data(const Eigen::MatrixXf& d);
    
    Eigen::MatrixXf get_grad() const;
    void set_grad(const Eigen::MatrixXf& g);

    bool get_requires_grad() const;

    // Pointer to underlying data (useful for binding to NumPy later)
    float* data_ptr();

    Tensor sum();
    Tensor mean();
    Tensor max();
    Tensor min();

    // Trigonometry
    Tensor sin();
    Tensor cos();

    // Operations
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator*(float scalar);
    Tensor operator/(const Tensor& other);
    

private:
   // The backend: Dynamic size, Float type
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;
    bool requires_grad = false; // Default to false
    std::vector<Tensor*> children;
    std::function<void(Tensor&)> backward_fn;
};

#endif // CORE_H