#include <torch/torch.h>
#include <vector>

at::Tensor shift_cuda_forward(
    at::Tensor input,at::Tensor xpos,at::Tensor ypos,const int stride);

std::vector<at::Tensor> shift_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor output,
    at::Tensor xpos,
    at::Tensor ypos,
    const int stride);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor shift_forward(
    at::Tensor input,at::Tensor xpos,at::Tensor ypos,const int stride) {
  CHECK_INPUT(input);
  return shift_cuda_forward(input,xpos,ypos,stride);
}

std::vector<at::Tensor> shift_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor output,
    at::Tensor xpos,
    at::Tensor ypos,
    const int stride) 
{
  CHECK_INPUT(grad_output);
  CHECK_INPUT(output);
  return shift_cuda_backward(
    grad_output,
    input,
    output,
    xpos,
    ypos,
    stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &shift_forward, "shift forward (CUDA)");
  m.def("backward", &shift_backward, "shift backward (CUDA)");
}
