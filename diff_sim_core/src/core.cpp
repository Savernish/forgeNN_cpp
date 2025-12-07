#include "core.h"
#include <iostream>
#include <unordered_set>

Tensor::Tensor(int rows, int cols, bool req_grad) {
    data.resize(rows, cols);
    data.setZero();
    set_requires_grad(req_grad);
}

Tensor::Tensor(int size, bool req_grad) {
    data.resize(size, 1);
    data.setZero();
    set_requires_grad(req_grad);
}

Tensor::Tensor(std::vector<float> data_list, bool req_grad) {
    data.resize(data_list.size(), 1);
    for (size_t i = 0; i < data_list.size(); ++i) {
        data(i, 0) = data_list[i];
    }
    set_requires_grad(req_grad);
}

void Tensor::set(int r, int c, float value) {
    if (r >= 0 && r < data.rows() && c >= 0 && c < data.cols()) {
        data(r, c) = value;
    }
}

float Tensor::get(int r, int c) const {
    if (r >= 0 && r < data.rows() && c >= 0 && c < data.cols()) {
        return data(r, c);
    }
    return 0.0f;
}

void Tensor::set_requires_grad(bool req) {
    this->requires_grad = req;
    if (req && grad.size() == 0) {
        grad.resizeLike(data);
        grad.setZero();
    }
}

void Tensor::zero_grad() {
    if (grad.size() > 0) {
        grad.setZero();
    }
}

void Tensor::backward() {
    if (!requires_grad) {
        std::cerr << "Warning: called backward() on a Tensor that does not require grad." << std::endl;
        return;
    }

    if (grad.size() == 0) {
        grad.resizeLike(data);
    }
    grad.setOnes();

    // Topological Sort
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    
    std::function<void(Tensor*)> recurse = [&](Tensor* node) {
        if (visited.find(node) == visited.end()) {
            visited.insert(node);
            for (Tensor* child : node->children) {
                recurse(child);
            }
            topo.push_back(node);
        }
    };
    recurse(this);

    // Backward Pass
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* node = *it;
        if (node->backward_fn) {
            node->backward_fn(*node);
        }
    }
}

int Tensor::rows() const { return data.rows(); }
int Tensor::cols() const { return data.cols(); }
float* Tensor::data_ptr() { return data.data(); }

Tensor Tensor::sum() {
    Tensor result(1, 1, false); // Scalar result
    result.data(0, 0) = this->data.sum();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);
        
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // Gradient of sum is 1 for all elements
                // distribute scalar gradient to all elements
                float grad_val = self.grad(0, 0); 
                this->grad.array() += grad_val;
            }
        };
    }
    return result;
}

Tensor Tensor::sin() {
    Tensor result(data.rows(), data.cols(), false);
    result.data = data.array().sin().matrix();

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);

        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dy * cos(x)
                this->grad.array() += self.grad.array() * this->data.array().cos();
            }
        };
    }
    return result;
}

Tensor Tensor::cos() {
    Tensor result(data.rows(), data.cols(), false);
    result.data = data.array().cos().matrix();

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);

        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dy * -sin(x)
                this->grad.array() -= self.grad.array() * this->data.array().sin();
            }
        };
    }
    return result;
}

Tensor Tensor::mean() {
    Tensor result(1, 1, false);
    result.data(0, 0) = this->data.mean();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);
        
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                this->grad.array() += self.grad.array() / this->data.rows();
            }
        };
    }
    return result;
}


// ---------------- Operators ----------------

Tensor Tensor::operator+(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data + other.data;

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad += self.grad;
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad += self.grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data - other.data;

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad += self.grad;
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad -= self.grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = (this->data.array() * other.data.array()).matrix();

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();

        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() * other.data.array();
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad.array() += self.grad.array() * this->data.array();
            }
        };
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = (this->data.array() / other.data.array()).matrix();

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();

        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() / other.data.array();
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad.array() -= self.grad.array() * this->data.array() / (other.data.array().square());
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(float scalar) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data * scalar;

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));

        result.backward_fn = [this, scalar](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dz * scalar
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() * scalar;
            }
        };
    }
    return result;
}

// Accessors
Eigen::MatrixXf Tensor::get_data() const { return data; }
void Tensor::set_data(const Eigen::MatrixXf& d) { data = d; }

Eigen::MatrixXf Tensor::get_grad() const { return grad; }
void Tensor::set_grad(const Eigen::MatrixXf& g) { grad = g; }

bool Tensor::get_requires_grad() const { return requires_grad; }
