#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "opencl_runtime.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "model.hpp"

namespace py = pybind11;

void test_print() {
    py::print("Hello from pybind11!", py::arg("flush")=true);
    std::cerr << "Hello from cerr!\n" << std::flush;
}

void initialize_opencl() {
    py::print("gaga(version:2.7)");

    opencl_runtime::getInstance().initialize();
    py::print("OpenCL Runtime Initialized", py::arg("flush")=true);

}

Tensor from_numpy(py::array_t<float> array) {
    py::buffer_info info = array.request();
    std::vector<size_t> shape(info.shape.begin(), info.shape.end());
    std::vector<float> data((float*)info.ptr, (float*)info.ptr + array.size());
    return Tensor(shape, data);
}

py::array_t<float> to_numpy(Tensor& tensor) {
    auto raw_shape = tensor.getShape();
    std::vector<py::ssize_t> py_shape;
    for (auto s : raw_shape) {
        py_shape.push_back(static_cast<py::ssize_t>(s));
    }

    // 注意：預設情況下 pybind11 會「複製」這份資料到 Python
    return py::array_t<float>(
        py_shape, 
        tensor.getDataRef().data()
    );
}

PYBIND11_MODULE(monke_frame, m) {
    py::add_ostream_redirect(m, "ostream_redirect");
    static py::scoped_ostream_redirect cout_redirect(
        std::cout,                                 // C++ stream
        py::module_::import("sys").attr("stdout")  // Python stream
    );
    static py::scoped_ostream_redirect cerr_redirect(
        std::cerr,
        py::module_::import("sys").attr("stderr")
    );

    m.def("initialize_opencl", &initialize_opencl, "初始化 OpenCL Runtime");
    m.def("test_print", &test_print);

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](py::array_t<float> arr) {
        return from_numpy(arr);
            }))
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        
        .def("getTotalSize", &Tensor::getTotalSize)
        .def("toGPU", py::overload_cast<>(&Tensor::toGPU))
        .def("toCPU", py::overload_cast<>(&Tensor::toCPU))
        .def("getData", py::overload_cast<>(&Tensor::getData, py::const_))
        .def("getShape", &Tensor::getShape)
        .def("getDataRef", &Tensor::getDataRef, py::return_value_policy::reference_internal)
        .def("set", &Tensor::set, py::arg("indices"), py::arg("value"))
        .def("get", &Tensor::get, py::arg("indices"))

        .def("print", &Tensor::print, py::arg("max_elements") = 10);
        

    m.def("from_numpy", &from_numpy);
    m.def("to_numpy", &to_numpy);
    
    // Layers
    py::class_<Layer>(m, "Layer");  // Base class (不會直接用到)

    py::class_<Scale, Layer>(m, "Scale")
        .def(py::init<size_t, float>(), py::arg("size"), py::arg("scale_factor") = 1.0f);
    py::class_<Dropout, Layer>(m, "Dropout")
        .def(py::init<size_t, float>(), py::arg("size"), py::arg("p"));

    py::class_<Dense, Layer>(m, "Dense")
        .def(py::init<size_t, size_t>(), py::arg("input_size"), py::arg("output_size"));

    py::class_<Convolution, Layer>(m, "Convolution")
        .def(py::init<size_t, size_t, size_t, size_t>(), py::arg("input_channels"), py::arg("input_size"), py::arg("output_channels"), py::arg("kernel_size"));

    py::class_<ReLU, Layer>(m, "ReLU")
        .def(py::init<size_t, float>(), py::arg("size"), py::arg("slope") = 0.0f);

    py::class_<Softmax, Layer>(m, "Softmax")
        .def(py::init<size_t>(), py::arg("size"));
    
    py::class_<BN, Layer>(m, "BN")
        .def(py::init<size_t, float>(), py::arg("size"), py::arg("p") = 0.1f);

    // Loss
    py::class_<Loss>(m, "Loss");  // Base 

    py::class_<MSE, Loss>(m, "MSE")
        .def(py::init<size_t>(), py::arg("size"));
    
    py::class_<CrossEntropy, Loss>(m, "CrossEntropy")
        .def(py::init<size_t>(), py::arg("size"));

    // Optimizers
    py::class_<Optimizer>(m, "Optimizer");  // Base

    py::class_<GD, Optimizer>(m, "GD")
        .def(py::init<>());

    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init<float, float, float>(), py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("epsilon") = 1e-8f);
    
    // Model
    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("add_layer", &Model::add_layer, py::arg("layer"), py::keep_alive<1, 2>())
        .def("compile", &Model::compile, py::arg("loss_function"), py::arg("optimizer"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("auto_batching", &Model::auto_batching, py::arg("inputs"), py::arg("max_batch_size"))
        .def("setIntermediateInputs", &Model::setIntermediateInputs, py::arg("max_batch_size(set to 0 to disable batching)"))
        .def("train", &Model::train, py::arg("inputs"), py::arg("targets"), py::arg("batch_size"), py::arg("learning_rate"))
        .def("predict", &Model::predict, py::arg("inputs"));
        

}
