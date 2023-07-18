#include "include/fused.h"
#include "include/common.h"


std::tuple<torch::Tensor,
           torch::Tensor> // (residual_output (FP), ln_output (INT8))
dq_add_layernorm_q(
    torch::Tensor input,          // INT32
    float input_scale,            // FP
    torch::Tensor residual_input, // FP
    torch::Tensor gamma,          // FP
    torch::Tensor beta,           // FP
    float epsilon                 // FP
    ) // The output scale has already been fused into gamma and beta
{
  // residual_output = residual_input + input * input_scale
  auto residual_output_fp = torch::add(residual_input, input, input_scale);

  auto ln_output_fp =
      torch::layer_norm(residual_output_fp, {residual_output_fp.size(-1)},
                        gamma, beta, epsilon);
  ln_output_fp.clamp_(-128, 127).round_();
  auto ln_output_int8 = ln_output_fp.to(torch::kChar);
  return std::make_tuple(residual_output_fp, ln_output_int8);
}


std::tuple<torch::Tensor, torch::Tensor> // (residual_output (FP), rn_output (INT8))
dq_add_rmsnorm_q(torch::Tensor input,          // INT32
                 float input_scale,            // FP
                 torch::Tensor residual_input, // FP
                 torch::Tensor gamma,          // FP
                 float epsilon                 // FP
) // The output scale has already been fused into gamma
{
  // residual_output = residual_input + input * input_scale
  auto residual_output_fp = torch::add(residual_input, input, input_scale);
  auto variance = torch::pow(residual_output_fp, 2.0);
  variance = torch::mean(variance, -1, true);
  variance = torch::add(variance, epsilon);
  variance = torch::rsqrt(variance);
  auto output_fp = torch::multiply(residual_output_fp, variance);
  output_fp = torch::multiply(output_fp, gamma);
  output_fp.clamp_(-128, 127).round_();
  auto output_int8 = output_fp.to(torch::kChar);
  return std::make_tuple(residual_output_fp, output_int8);
}