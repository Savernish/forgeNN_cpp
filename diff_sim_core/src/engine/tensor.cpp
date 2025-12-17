#include "engine/tensor.h"
#include <iostream>
#include <unordered_set>

Tensor::Tensor() {
    m_Data.resize(0, 0);
    SetRequiresGrad(false);
}

Tensor::Tensor(int rows, int cols, bool requiresGrad) {
    m_Data.resize(rows, cols);
    m_Data.setZero();
    SetRequiresGrad(requiresGrad);
}

Tensor::Tensor(int size, bool requiresGrad) {
    m_Data.resize(size, 1);
    m_Data.setZero();
    SetRequiresGrad(requiresGrad);
}

Tensor::Tensor(std::vector<float> dataList, bool requiresGrad) {
    m_Data.resize(dataList.size(), 1);
    for (size_t i = 0; i < dataList.size(); ++i) {
        m_Data(i, 0) = dataList[i];
    }
    SetRequiresGrad(requiresGrad);
}

void Tensor::Set(int r, int c, float value) {
    if (r >= 0 && r < m_Data.rows() && c >= 0 && c < m_Data.cols()) {
        m_Data(r, c) = value;
    }
}

float Tensor::Get(int r, int c) const {
    if (r >= 0 && r < m_Data.rows() && c >= 0 && c < m_Data.cols()) {
        return m_Data(r, c);
    }
    return 0.0f;
}

void Tensor::SetRequiresGrad(bool requiresGrad) {
    this->m_bRequiresGrad = requiresGrad;
    if (requiresGrad && m_Grad.size() == 0) {
        m_Grad.resizeLike(m_Data);
        m_Grad.setZero();
    }
}

void Tensor::ZeroGrad() {
    if (m_Grad.size() > 0) {
        m_Grad.setZero();
    }
}

void Tensor::Backward() {
    if (!m_bRequiresGrad) {
        std::cerr << "Warning: called Backward() on a Tensor that does not require grad." << std::endl;
        return;
    }

    if (m_Grad.size() == 0) {
        m_Grad.resizeLike(m_Data);
    }
    m_Grad.setOnes();

    // Iterative Topological Sort to avoid Stack Overflow
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    std::vector<Tensor*> stack;
    
    std::unordered_set<Tensor*> expanded;
    stack.push_back(this);
    
    while (!stack.empty()) {
        Tensor* pNode = stack.back();
        
        if (visited.count(pNode)) {
            stack.pop_back();
            continue;
        }
        
        if (expanded.count(pNode)) {
            visited.insert(pNode);
            topo.push_back(pNode);
            stack.pop_back();
        } else {
            expanded.insert(pNode);
            for (Tensor* pChild : pNode->m_Children) {
                if (visited.find(pChild) == visited.end()) {
                    stack.push_back(pChild);
                }
            }
        }
    }
    

    // Backward Pass
    int count = 0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* pNode = *it;
        if (pNode->m_BackwardFn) {
            pNode->m_BackwardFn(*pNode);
        }
    }
}

int Tensor::Rows() const { return m_Data.rows(); }
int Tensor::Cols() const { return m_Data.cols(); }
float* Tensor::DataPtr() { return m_Data.data(); }

// Reductions

// Sum (Scalar)
Tensor Tensor::Sum() {
    Tensor result(1, 1, false);
    result.m_Data(0, 0) = this->m_Data.sum();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(this);
        
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                float gradVal = self.m_Grad(0, 0); 
                this->m_Grad.array() += gradVal;
            }
        };
    }
    return result;
}

// Min (Scalar)
Tensor Tensor::Min() {
    Tensor result(1, 1, false);
    Eigen::Index r, c;
    float val = this->m_Data.minCoeff(&r, &c);
    result.m_Data(0,0) = val;

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this, r, c](Tensor& self) {
             if (this->m_bRequiresGrad) {
                 this->m_Grad(r, c) += self.m_Grad(0,0);
             }
        };
    }
    return result;
}

// Max (Scalar)
Tensor Tensor::Max() {
    Tensor result(1, 1, false);
    Eigen::Index r, c;
    float val = this->m_Data.maxCoeff(&r, &c);
    result.m_Data(0,0) = val;

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this, r, c](Tensor& self) {
             if (this->m_bRequiresGrad) {
                 this->m_Grad(r, c) += self.m_Grad(0,0);
             }
        };
    }
    return result;
}

// Sum (Axis)
Tensor Tensor::Sum(int axis) {
    if (axis != 0 && axis != 1) throw std::runtime_error("Axis must be 0 or 1");

    Tensor result(0,0);
    if (axis == 0) {
        result = Tensor(1, m_Data.cols(), false);
        result.m_Data = m_Data.colwise().sum();
    } else {
        result = Tensor(m_Data.rows(), 1, false);
        result.m_Data = m_Data.rowwise().sum();
    }

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this, axis](Tensor& self) {
            if (this->m_bRequiresGrad) {
                if (axis == 0) {
                    this->m_Grad.array() += self.m_Grad.array().row(0).replicate(this->Rows(), 1);
                } else {
                    this->m_Grad.array() += self.m_Grad.array().col(0).replicate(1, this->Cols());
                }
            }
        };
    }
    return result;
}

// Mean (Axis)
Tensor Tensor::Mean(int axis) {
    if (axis != 0 && axis != 1) throw std::runtime_error("Axis must be 0 or 1");
    Tensor result(0,0);
    if (axis == 0) {
        result = Tensor(1, m_Data.cols(), false);
        result.m_Data = m_Data.colwise().mean();
    } else {
        result = Tensor(m_Data.rows(), 1, false);
        result.m_Data = m_Data.rowwise().mean();
    }

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this, axis](Tensor& self) {
            if (this->m_bRequiresGrad) {
                float n = (axis == 0) ? (float)this->Rows() : (float)this->Cols();
                if (axis == 0) {
                     this->m_Grad.array() += (self.m_Grad.array().row(0) / n).replicate(this->Rows(), 1);
                } else {
                     this->m_Grad.array() += (self.m_Grad.array().col(0) / n).replicate(1, this->Cols());
                }
            }
        };
    }
    return result;
}

Tensor Tensor::Sin() {
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    result.m_Data = m_Data.array().sin().matrix();

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(this);

        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += self.m_Grad.array() * this->m_Data.array().cos();
            }
        };
    }
    return result;
}

Tensor Tensor::Cos() {
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    result.m_Data = m_Data.array().cos().matrix();

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(this);

        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() -= self.m_Grad.array() * this->m_Data.array().sin();
            }
        };
    }
    return result;
}

Tensor Tensor::Mean() {
    Tensor result(1, 1, false);
    result.m_Data(0, 0) = this->m_Data.mean();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(this);
        
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += self.m_Grad.array() / this->m_Data.rows();
            }
        };
    }
    return result;
}

Tensor Tensor::Select(int idx) const {
    Tensor result(1, 1, false);
    if (idx < 0 || idx >= m_Data.size()) {
        std::cerr << "Error: Index " << idx << " out of bounds for Tensor size " << m_Data.size() << std::endl;
        return result; 
    }
    result.m_Data(0,0) = this->m_Data(idx);
    
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(const_cast<Tensor*>(this));
        
        Tensor* pSelf = const_cast<Tensor*>(this);
        result.m_BackwardFn = [pSelf, idx](Tensor& self) {
            if (pSelf->m_bRequiresGrad) {
                pSelf->m_Grad(idx) += self.m_Grad(0,0);
            }
        };
    }
    return result;
}

Tensor Tensor::Stack(const std::vector<Tensor*>& tensors) {
    int n = tensors.size();
    if (n == 0) return Tensor(0, 1);
    
    Tensor result(n, 1, false); 
    bool bAnyGrad = false;
    for(auto* pTensor : tensors) {
        if(pTensor->GetRequiresGrad()) bAnyGrad = true;
    }
    
    if (bAnyGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        
        for(int i=0; i<n; ++i) {
            result.m_Data(i, 0) = tensors[i]->m_Data(0,0);
            if(tensors[i]->GetRequiresGrad()) {
                result.m_Children.push_back(tensors[i]);
            }
        }
        
        std::vector<Tensor*> inputs = tensors;
        
        result.m_BackwardFn = [inputs](Tensor& self) {
            for(size_t i=0; i<inputs.size(); ++i) {
                Tensor* pInput = inputs[i];
                if(pInput->GetRequiresGrad()) {
                    pInput->m_Grad(0,0) += self.m_Grad(i,0);
                }
            }
        };
    } else {
         for(int i=0; i<n; ++i) {
            result.m_Data(i, 0) = tensors[i]->m_Data(0,0);
        }
    }
    return result;
}


// ---------------- Operators ----------------

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    result.m_Data = this->m_Data + other.m_Data;

    if (this->m_bRequiresGrad || other.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(const_cast<Tensor*>(this));
        result.m_Children.push_back(const_cast<Tensor*>(&other));

        result.m_BackwardFn = [this, &other](Tensor& self) {
            if (this->m_bRequiresGrad) {
                const_cast<Tensor*>(this)->m_Grad += self.m_Grad;
            }
            if (other.m_bRequiresGrad) {
                const_cast<Tensor*>(&other)->m_Grad += self.m_Grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    result.m_Data = this->m_Data - other.m_Data;

    if (this->m_bRequiresGrad || other.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(const_cast<Tensor*>(this));
        result.m_Children.push_back(const_cast<Tensor*>(&other));

        result.m_BackwardFn = [this, &other](Tensor& self) {
            if (this->m_bRequiresGrad) {
                const_cast<Tensor*>(this)->m_Grad += self.m_Grad;
            }
            if (other.m_bRequiresGrad) {
                const_cast<Tensor*>(&other)->m_Grad -= self.m_Grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    bool bScalarBroadcast = (other.Rows() == 1 && other.Cols() == 1);
    
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    
    if (bScalarBroadcast) {
        result.m_Data = this->m_Data * other.m_Data(0, 0);
    } else {
        if (m_Data.rows() != other.Rows() || m_Data.cols() != other.Cols()) {
            throw std::runtime_error("Dimension mismatch in operator* " + 
                std::to_string(m_Data.rows()) + "x" + std::to_string(m_Data.cols()) + " vs " +
                std::to_string(other.Rows()) + "x" + std::to_string(other.Cols()));
        }
        result.m_Data = (this->m_Data.array() * other.m_Data.array()).matrix();
    }

    if (this->m_bRequiresGrad || other.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();

        result.m_Children.push_back(const_cast<Tensor*>(this));
        result.m_Children.push_back(const_cast<Tensor*>(&other));

        result.m_BackwardFn = [this, &other, bScalarBroadcast](Tensor& self) {
            if (this->m_bRequiresGrad) {
                if (bScalarBroadcast) {
                     const_cast<Tensor*>(this)->m_Grad.array() += self.m_Grad.array() * other.m_Data(0,0);
                } else {
                     const_cast<Tensor*>(this)->m_Grad.array() += self.m_Grad.array() * other.m_Data.array();
                }
            }
            if (other.m_bRequiresGrad) {
                if (bScalarBroadcast) {
                    float gradScalar = (self.m_Grad.array() * this->m_Data.array()).sum();
                    const_cast<Tensor*>(&other)->m_Grad(0,0) += gradScalar;
                } else {
                     const_cast<Tensor*>(&other)->m_Grad.array() += self.m_Grad.array() * this->m_Data.array();
                }
            }
        };
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    bool bScalarBroadcast = (other.Rows() == 1 && other.Cols() == 1);

    Tensor result(m_Data.rows(), m_Data.cols(), false);
    
    if (bScalarBroadcast) {
        result.m_Data = this->m_Data / other.m_Data(0, 0);
    } else {
         if (m_Data.rows() != other.Rows() || m_Data.cols() != other.Cols()) {
            throw std::runtime_error("Dimension mismatch in operator/ " + 
                std::to_string(m_Data.rows()) + "x" + std::to_string(m_Data.cols()) + " vs " +
                std::to_string(other.Rows()) + "x" + std::to_string(other.Cols()));
        }
        result.m_Data = (this->m_Data.array() / other.m_Data.array()).matrix();
    }

    if (this->m_bRequiresGrad || other.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();

        result.m_Children.push_back(const_cast<Tensor*>(this));
        result.m_Children.push_back(const_cast<Tensor*>(&other));

        result.m_BackwardFn = [this, &other, bScalarBroadcast](Tensor& self) {
            if (this->m_bRequiresGrad) {
                if (bScalarBroadcast) {
                    const_cast<Tensor*>(this)->m_Grad.array() += self.m_Grad.array() / other.m_Data(0,0);
                } else {
                    const_cast<Tensor*>(this)->m_Grad.array() += self.m_Grad.array() / other.m_Data.array();
                }
            }
            if (other.m_bRequiresGrad) {
                if (bScalarBroadcast) {
                     float s = other.m_Data(0,0);
                     float gradScalar = (self.m_Grad.array() * this->m_Data.array() * (-1.0f / (s*s))).sum();
                     const_cast<Tensor*>(&other)->m_Grad(0,0) += gradScalar;
                } else {
                    const_cast<Tensor*>(&other)->m_Grad.array() -= self.m_Grad.array() * this->m_Data.array() / (other.m_Data.array().square());
                }
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(m_Data.rows(), m_Data.cols(), false);
    result.m_Data = this->m_Data * scalar;

    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.resizeLike(result.m_Data);
        result.m_Grad.setZero();
        
        result.m_Children.push_back(const_cast<Tensor*>(this));

        result.m_BackwardFn = [this, scalar](Tensor& self) {
            if (this->m_bRequiresGrad) {
                const_cast<Tensor*>(this)->m_Grad.array() += self.m_Grad.array() * scalar;
            }
        };
    }
    return result;
}

// Mathematical Functions

// Transpose
Tensor Tensor::Transpose() {
    Tensor result(this->m_Data.cols(), this->m_Data.rows(), false);
    result.m_Data = this->m_Data.transpose();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad += self.m_Grad.transpose();
            }
        };
    }
    return result;
}

// Power
Tensor Tensor::Pow(float exponent) {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.array().pow(exponent);
    
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        
        result.m_BackwardFn = [this, exponent](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += exponent * this->m_Data.array().pow(exponent - 1.0f) * self.m_Grad.array();
            }
        };
    }
    return result;
}

// Exp
Tensor Tensor::Exp() {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.array().exp();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += self.m_Data.array() * self.m_Grad.array(); 
            }
        };
    }
    return result;
}

// Log
Tensor Tensor::Log() {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.array().log();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += self.m_Grad.array() / this->m_Data.array();
            }
        };
    }
    return result;
}

// Sqrt
Tensor Tensor::Sqrt() {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.array().sqrt();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += 0.5f * self.m_Grad.array() / self.m_Data.array();
            }
        };
    }
    return result;
}

// Abs
Tensor Tensor::Abs() {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.array().abs();
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                this->m_Grad.array() += self.m_Grad.array() * this->m_Data.array().sign();
            }
        };
    }
    return result;
}

// Clamp
Tensor Tensor::Clamp(float minVal, float maxVal) {
    Tensor result(this->m_Data.rows(), this->m_Data.cols(), false);
    result.m_Data = this->m_Data.cwiseMax(minVal).cwiseMin(maxVal);
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        result.m_BackwardFn = [this, minVal, maxVal](Tensor& self) {
            if (this->m_bRequiresGrad) {
                Eigen::ArrayXXf x = this->m_Data.array();
                this->m_Grad.array() += self.m_Grad.array() * (x >= minVal && x <= maxVal).cast<float>();
            }
        };
    }
    return result;
}


// Reshape
Tensor Tensor::Reshape(int r, int c) {
    if (r * c != this->m_Data.size()) {
         std::cerr << "Error: Reshape size mismatch. Total " << this->m_Data.size() << " requested " << r << "x" << c << std::endl;
         return Tensor(1,1);
    }
    
    Tensor result(r, c, false);
    result.m_Data = Eigen::Map<Eigen::MatrixXf>(this->m_Data.data(), r, c);
    
    if (this->m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(this);
        
        result.m_BackwardFn = [this](Tensor& self) {
            if (this->m_bRequiresGrad) {
                Eigen::Map<Eigen::VectorXf> flatGrad(this->m_Grad.data(), this->m_Grad.size());
                Eigen::Map<Eigen::VectorXf> flatSelf(self.m_Grad.data(), self.m_Grad.size());
                flatGrad += flatSelf;
            }
        };
    }
    return result;
}

// Concatenate
Tensor Tensor::Cat(const std::vector<Tensor*>& tensors, int dim) {
    if (tensors.empty()) return Tensor(0, 0);

    int rows = tensors[0]->m_Data.rows();
    int cols = tensors[0]->m_Data.cols();
    
    int totalRows = 0;
    int totalCols = 0;
    
    if (dim == 0) {
        totalCols = cols;
        for (const auto* pTensor : tensors) {
            if (pTensor->m_Data.cols() != cols) throw std::runtime_error("Dimension mismatch in Cat(dim=0)");
            totalRows += pTensor->m_Data.rows();
        }
    } else if (dim == 1) {
        totalRows = rows;
        for (const auto* pTensor : tensors) {
            if (pTensor->m_Data.rows() != rows) throw std::runtime_error("Dimension mismatch in Cat(dim=1)");
            totalCols += pTensor->m_Data.cols();
        }
    } else {
        throw std::runtime_error("Invalid dimension for Cat (only 0 or 1 supported)");
    }

    Tensor result(totalRows, totalCols, false);
    
    int currentOffset = 0;
    for (const auto* pTensor : tensors) {
        if (dim == 0) {
            result.m_Data.block(currentOffset, 0, pTensor->m_Data.rows(), cols) = pTensor->m_Data;
            currentOffset += pTensor->m_Data.rows();
        } else {
            result.m_Data.block(0, currentOffset, rows, pTensor->m_Data.cols()) = pTensor->m_Data;
            currentOffset += pTensor->m_Data.cols();
        }
        
        if (pTensor->GetRequiresGrad()) {
            result.SetRequiresGrad(true);
            result.m_Children.push_back(const_cast<Tensor*>(pTensor));
        }
    }
    
    if (result.GetRequiresGrad()) {
        result.m_Grad.setZero();
        
        std::vector<Tensor*> inputs = tensors;
        
        result.m_BackwardFn = [inputs, dim](Tensor& self) {
            int offset = 0;
            for (Tensor* pTensor : inputs) {
                int r = pTensor->m_Data.rows();
                int c = pTensor->m_Data.cols();
                
                if (pTensor->GetRequiresGrad()) {
                    if (dim == 0) {
                        pTensor->m_Grad += self.m_Grad.block(offset, 0, r, c);
                    } else {
                        pTensor->m_Grad += self.m_Grad.block(0, offset, r, c);
                    }
                }
                
                if (dim == 0) offset += r;
                else offset += c;
            }
        };
    }
    
    return result;
}


// Matrix Multiplication
Tensor Tensor::Matmul(const Tensor& other) {
    if (this->m_Data.cols() != other.m_Data.rows()) {
        throw std::runtime_error("Shape mismatch for Matmul");
    }
    Tensor result(this->m_Data.rows(), other.m_Data.cols(), false);
    result.m_Data = this->m_Data * other.m_Data;
    if (this->m_bRequiresGrad || other.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        if (this->m_bRequiresGrad) result.m_Children.push_back((Tensor*)this);
        if (other.m_bRequiresGrad) result.m_Children.push_back((Tensor*)&other);
        Tensor* pA = (Tensor*)this;
        Tensor* pB = (Tensor*)&other;
        result.m_BackwardFn = [pA, pB](Tensor& self) {
            if (pA->GetRequiresGrad()) {
                pA->m_Grad += self.m_Grad * pB->m_Data.transpose();
            }
            if (pB->GetRequiresGrad()) {
                pB->m_Grad += pA->m_Data.transpose() * self.m_Grad;
            }
        };
    }
    return result;
}


// Gaussian log probability for policy gradients
// log π(a|μ,σ) = -0.5 × ((a - μ)/σ)² - log(σ) - 0.5×log(2π)
Tensor Tensor::GaussianLogProb(const Tensor& action, const Tensor& mean, const Tensor& logStd) {
    const float LOG_2PI = 1.8378770664093453f;
    
    Tensor result(1, 1, false);
    float total = 0.0f;
    int n = action.Rows();
    
    for (int i = 0; i < n; i++) {
        float a = action.m_Data(i, 0);
        float mu = mean.m_Data(i, 0);
        float logS = logStd.m_Data(i, 0);
        float s = std::exp(logS);
        float diff = (a - mu) / s;
        total += -0.5f * diff * diff - logS - 0.5f * LOG_2PI;
    }
    result.m_Data(0, 0) = total;
    
    if (mean.m_bRequiresGrad || logStd.m_bRequiresGrad) {
        result.SetRequiresGrad(true);
        result.m_Grad.setZero();
        result.m_Children.push_back(const_cast<Tensor*>(&mean));
        result.m_Children.push_back(const_cast<Tensor*>(&logStd));
        
        Tensor* pAction = const_cast<Tensor*>(&action);
        Tensor* pMean = const_cast<Tensor*>(&mean);
        Tensor* pLogStd = const_cast<Tensor*>(&logStd);
        int numDims = n;
        
        result.m_BackwardFn = [pAction, pMean, pLogStd, numDims](Tensor& self) {
            for (int i = 0; i < numDims; i++) {
                float a = pAction->m_Data(i, 0);
                float mu = pMean->m_Data(i, 0);
                float logS = pLogStd->m_Data(i, 0);
                float s = std::exp(logS);
                float diff = a - mu;
                
                if (pMean->m_bRequiresGrad) {
                    pMean->m_Grad(i, 0) += self.m_Grad(0, 0) * diff / (s * s);
                }
                if (pLogStd->m_bRequiresGrad) {
                    float normalizedDiff = diff / s;
                    pLogStd->m_Grad(i, 0) += self.m_Grad(0, 0) * (normalizedDiff * normalizedDiff - 1.0f);
                }
            }
        };
    }
    return result;
}

// Accessors
Eigen::MatrixXf Tensor::GetData() const { return m_Data; }
void Tensor::SetData(const Eigen::MatrixXf& d) { m_Data = d; }

Eigen::MatrixXf Tensor::GetGrad() const { return m_Grad; }
void Tensor::SetGrad(const Eigen::MatrixXf& g) { m_Grad = g; }

bool Tensor::GetRequiresGrad() const { return m_bRequiresGrad; }
