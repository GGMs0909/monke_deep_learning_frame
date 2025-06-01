#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

class opencl_runtime {
private:
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device; // �x�s��w���]��
    cl::Program program;
    cl_int err; // �O�d cl_int ���~�X�ˬd

    bool is_initialized_flag; // �l�ܬO�_�w���\��l��

    // �p���غc�l�A����~�������Ыع��
    opencl_runtime();

    // �p���Ѻc�l�A�Ω��Ҫ��M�z
    ~opencl_runtime();

    // �T�Ϋ����غc�l�M������ȹB��l�A�T�O��Ұߤ@��
    opencl_runtime(const opencl_runtime&) = delete;
    opencl_runtime& operator=(const opencl_runtime&) = delete;

    // ���U�禡�G�ˬd OpenCL ���~�ðh�X
    void check_error(cl_int err, const char* operation);

public:
    // �����ҹ��
    static opencl_runtime& getInstance();

    // **�s�W�G�㦡��l�Ƥ�k**
    void initialize();

    // ����W�U��
    cl::Context& get_context();

    // ����{��
    cl::Program& get_program();

    // ����R�O���C
    cl::CommandQueue& get_queue();

    // ����]��
    cl::Device& get_device();
};