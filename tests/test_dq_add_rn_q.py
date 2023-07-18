import torch
from icecream import ic
from torch_int.functional.fused import dq_add_rmsnorm_q_py, dq_add_rmsnorm_q_cpp


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


@torch.no_grad()
def test_dq_add_rmsnorm_q():
    B, L, H = 2, 3, 4
    input_int32 = torch.randint(-65536, 65536, (B, L, H), dtype=torch.int32)
    input_scale_fp = 0.01
    residual_input_fp = torch.randn(B, L, H)
    rmsnorm = LlamaRMSNorm(H)
    gamma = rmsnorm.weight
    eps = rmsnorm.variance_epsilon
    py_output = dq_add_rmsnorm_q_py(
        input_int32, input_scale_fp, residual_input_fp, gamma, eps)
    ic(py_output)
    cpp_output = dq_add_rmsnorm_q_cpp(
        input_int32, input_scale_fp, residual_input_fp, gamma, eps)
    ic(cpp_output)
    ic(torch.allclose(py_output[0], cpp_output[0]))
    ic(torch.allclose(py_output[1], cpp_output[1]))


if __name__ == '__main__':
    test_dq_add_rmsnorm_q()