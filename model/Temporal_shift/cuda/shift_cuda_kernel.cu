#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <iostream>

namespace {
  template <typename scalar_t>
  __global__ void shift_cuda_forward_kernel(
      const scalar_t* __restrict__ input,
      scalar_t* output,
      scalar_t* xpos, 
      scalar_t* ypos,
      const int batch,
      const int channel,
      const int bottom_height,
      const int bottom_width,
      const int top_height,
      const int top_width,
      const int stride) 
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;


    if (index < batch*channel*top_height*top_width)
    {
	    const int top_sp_dim = top_height * top_width;
	    const int bottom_sp_dim = bottom_height * bottom_width;
	    const int n = index/(channel * top_sp_dim);       
	    const int idx = index%(channel * top_sp_dim);     
	    const int c_out = idx/top_sp_dim;                     
	    const int c_in = c_out;                              
	    const int sp_idx = idx%top_sp_dim;                 
	    const int h = sp_idx/top_width;               
	    const int w = sp_idx%top_width;            
	    const scalar_t* data_im_ptr = input + n*channel*bottom_sp_dim + c_in*bottom_sp_dim; // ->(n,c) 

	    const int h_offset = h * stride;             // h on input feature map
	    const int w_offset = w;              // w on input feature map

	    scalar_t val = 0;
	    const scalar_t x = xpos[c_in];
	    const scalar_t y = ypos[c_in];

	    int h_im, w_im;
	    int x1 = floorf(x);
	    int x2 = x1+1;
	    int y1 = floorf(y);
	    int y2 = y1+1;

	    h_im = h_offset + y1;
	    w_im = w_offset + x1;
	    scalar_t q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

	    h_im = h_offset + y1;
	    w_im = w_offset + x2;
	    scalar_t q21 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

	    h_im = h_offset + y2;
	    w_im = w_offset + x1;
	    scalar_t q12 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

	    h_im = h_offset + y2;
	    w_im = w_offset + x2;
	    scalar_t q22 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

	    scalar_t dx = x-x1;
	    scalar_t dy = y-y1;

	    val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
	    output[index] = val;
	}
  }

  template <typename scalar_t>
  __global__ void Shift_Bottom_Backward_Stride1(
        const scalar_t* __restrict__ grad_output,
        scalar_t* grad_input,
        scalar_t* xpos,
        scalar_t* ypos,
        const int batch,
        const int channel,
        const int bottom_height,
        const int bottom_width)  
  {
      const int index = blockIdx.x * blockDim.x + threadIdx.x;

	  if (index < batch*channel*bottom_height*bottom_width)
	  {
	      const int top_sp_dim = bottom_height * bottom_width;                // h*w
	      const int bottom_sp_dim = bottom_height * bottom_width;   
	      const int n = index/(channel * bottom_sp_dim);    
	      const int idx = index%(channel * bottom_sp_dim);
	      const int c_in = idx/bottom_sp_dim;
	      const int c_out = c_in;
	      const int sp_idx = idx%bottom_sp_dim;
	      const int h_col = sp_idx/bottom_width;
	      const int w_col = sp_idx%bottom_width;
	      const scalar_t* top_diff_ptr = grad_output + n*channel*top_sp_dim + c_out*top_sp_dim;

	      const int h_offset = h_col;
	      const int w_offset = w_col;

	      scalar_t val = 0;
	      const scalar_t x = -xpos[c_in];  //reverse position
	      const scalar_t y = -ypos[c_in];

	      int h_im, w_im;

	      int x1 = floorf(x);
	      int x2 = x1+1;
	      int y1 = floorf(y);
	      int y2 = y1+1;

	      //q11
	      scalar_t q11 = 0;

	      h_im = (h_offset + y1);
	      w_im = (w_offset + x1);
	      q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? top_diff_ptr[h_im*bottom_width + w_im] : 0;

	      //q21
	      scalar_t q21 = 0;

	      h_im = (h_offset + y1);
	      w_im = (w_offset + x2);
	      q21 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? top_diff_ptr[h_im*bottom_width + w_im] : 0;

	      //q12
	      scalar_t q12 = 0;

	      h_im = (h_offset + y2);
	      w_im = (w_offset + x1);
	      q12 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? top_diff_ptr[h_im*bottom_width + w_im] : 0;

	      //q22
	      scalar_t q22 = 0;

	      h_im = (h_offset + y2);
	      w_im = (w_offset + x2);
	      q22 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? top_diff_ptr[h_im*bottom_width + w_im] : 0;

	      scalar_t dx = x-x1;
	      scalar_t dy = y-y1;

	      val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
	      grad_input[index] = val;
	}
  } 


  template <typename scalar_t>
  __global__ void Shift_Bottom_Backward(
        const scalar_t* __restrict__ grad_output,
        scalar_t* grad_input,
        scalar_t* xpos,
        scalar_t* ypos,
        const int batch,
        const int channel,
        const int bottom_height,
        const int bottom_width)  
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;


    if (index < batch*channel*bottom_height*bottom_width)
    {

	    const int top_height = bottom_height/2;
	    const int top_width = bottom_width;
	    const int stride = 2;
	    const int top_sp_dim = top_height * top_width;
	    const int bottom_sp_dim = bottom_height * bottom_width;
	    const int n = index/(channel * bottom_sp_dim);
	    const int idx = index%(channel * bottom_sp_dim);
	    const int c_in = idx/bottom_sp_dim;
	    const int c_out = c_in;
	    const int sp_idx = idx%bottom_sp_dim;
	    const int h_col = sp_idx/bottom_width;
	    const int w_col = sp_idx%bottom_width;
	    const scalar_t* top_diff_ptr = grad_output + n*channel*top_sp_dim + c_out*top_sp_dim;

	    const int h_offset = h_col;
	    const int w_offset = w_col;


	    scalar_t val = 0;
	    const scalar_t x = -xpos[c_in]; 
	    const scalar_t y = -ypos[c_in];

	    int h_im, w_im;
	    int x1 = floorf(x);
	    int x2 = x1+1;
	    int y1 = floorf(y);
	    int y2 = y1+1;

	    //q11
	    scalar_t q11 = 0;

	    h_im = (h_offset + y1);
	    w_im = (w_offset + x1);
	    if(h_im%stride == 0)
	    {
	      h_im=h_im/stride;

	      q11 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
	    }

	    //q21
	    scalar_t q21 = 0;

	    h_im = (h_offset + y1);
	    w_im = (w_offset + x2);
	    if(h_im%stride == 0)
	    {
	      h_im=h_im/stride;

	      q21 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
	    }

	    //q12
	    scalar_t q12 = 0;

	    h_im = (h_offset + y2);
	    w_im = (w_offset + x1);

	    if(h_im%stride == 0)
	    {
	      h_im=h_im/stride;

	      q12 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
	    }

	    //q22
	    scalar_t q22 = 0;

	    h_im = (h_offset + y2);
	    w_im = (w_offset + x2);

	    if(h_im%stride == 0)
	    {
	      h_im=h_im/stride;

	      q22 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
	    }

	    scalar_t dx = x-x1;
	    scalar_t dy = y-y1;

	    val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
	    grad_input[index] = val;
	}
  } // namespace



  template <typename scalar_t>
  __inline__ __device__ void myAtomicAdd(scalar_t *buf, scalar_t val);

  template <>
  __inline__ __device__ void myAtomicAdd<float>(float *buf, float val)
  {
    atomicAdd(buf, val);
  }

  template <>
  __inline__ __device__ void myAtomicAdd<double>(double *buf, double val)
  {
    //Not Supported
  }



  template <typename scalar_t>
  __global__ void Shift_Position_Backward(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ grad_output,
        scalar_t* grad_input,
        scalar_t* xpos,
        scalar_t* ypos,
        scalar_t* grad_xpos_bchw,
        scalar_t* grad_ypos_bchw,
        const int batch,
        const int channel,
        const int bottom_height,
        const int bottom_width,
        const int stride)  
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int top_height = bottom_height/stride;
    const int top_width = bottom_width;


    if (index < batch*channel*top_height*top_width)
    {
	    const int top_sp_dim = top_height * top_width;
	    const int bottom_sp_dim = bottom_height * bottom_width;
	    const int n = index/(channel * top_sp_dim);
	    const int idx = index%(channel * top_sp_dim);
	    const int c_mul = 1;
	    const int c_out = idx/top_sp_dim;
	    const int c_in = c_out/c_mul;
	    const int sp_idx = idx%top_sp_dim;
	    const int h = sp_idx/top_width;
	    const int w = sp_idx%top_width;
	    const scalar_t* data_im_ptr = input + n*channel*bottom_sp_dim + c_in*bottom_sp_dim;

	    const int h_offset = h * stride;
	    const int w_offset = w;

	    //output : 2*(C) x (1*H*W)
	    const int kernel_offset = top_sp_dim;
	    const int c_off = c_out % c_mul;

	    scalar_t val_x = 0, val_y = 0;

	    const scalar_t shiftX = xpos[c_in];
	    const scalar_t shiftY = ypos[c_in];


	    const int ix1 = floorf(shiftX);
	    const int ix2 = ix1+1;
	    const int iy1 = floorf(shiftY);
	    const int iy2 = iy1+1;
	    const scalar_t dx = shiftX-ix1;
	    const scalar_t dy = shiftY-iy1;

	    const int h_im1 = h_offset + iy1;
	    const int h_im2 = h_offset + iy2;

	    const int w_im1 = w_offset + ix1;
	    const int w_im2 = w_offset + ix2;

	    const scalar_t q11 = (h_im1 >= 0 && w_im1 >= 0 && h_im1 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im1] : 0;
	    const scalar_t q21 = (h_im1 >= 0 && w_im2 >= 0 && h_im1 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im2] : 0;
	    const scalar_t q12 = (h_im2 >= 0 && w_im1 >= 0 && h_im2 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im1] : 0;
	    const scalar_t q22 = (h_im2 >= 0 && w_im2 >= 0 && h_im2 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im2] : 0;

	    val_x = (1-dy)*(q21-q11)+dy*(q22-q12);
	    val_y = (1-dx)*(q12-q11)+dx*(q22-q21);



	  	grad_xpos_bchw[index] = val_x * grad_output[index];
	    grad_ypos_bchw[index] = val_y * grad_output[index];

	  	//grad_xpos_bchw[index] = val_x;
	    //grad_ypos_bchw[index] = val_y;

	    //grad_xpos_bchw[index] = 0;
	    //grad_ypos_bchw[index] = 0;

	    //scalar_t* out_ptr_x = grad_xpos_bchw + index;
	    //scalar_t* out_ptr_y = grad_ypos_bchw + index;

	    //myAtomicAdd(out_ptr_x, val_x * grad_output[index]);
	    //myAtomicAdd(out_ptr_y, val_y * grad_output[index]);
	}
  } // namespace






  template <typename scalar_t>
  __global__ void applyShiftConstraint(
        scalar_t* grad_xpos,
        scalar_t* grad_ypos,
        const int channel)  
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < channel)
    {
	  const scalar_t dx = grad_xpos[index];
	  const scalar_t dy = grad_ypos[index];
	  const scalar_t dr = sqrt(dy*dy);

	  if(dr!=0)
	  {
		  grad_xpos[index] = dx/dr*0.0;
		  grad_ypos[index] = dy/dr*0.01;
	  }
	  else                                  // without this, the grad_ypos may be large.
	  {
		  grad_xpos[index] = 0.0;
		  grad_ypos[index] = 0.0001;
	  }
	}
  } // namespace




}




at::Tensor shift_cuda_forward(
    at::Tensor input,at::Tensor xpos,at::Tensor ypos,const int stride) {

  auto output = at::zeros({input.size(0), input.size(1), input.size(2)/stride, input.size(3)}, input.options());

  const dim3 blocks((input.size(0)*input.size(1)*input.size(2)*input.size(3)/stride+1024-1)/1024);
  const int threads = 1024;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "shift_forward_cuda", ([&] {
    shift_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.data<scalar_t>(),
      output.data<scalar_t>(),
      xpos.data<scalar_t>(),
      ypos.data<scalar_t>(),
      input.size(0),
      input.size(1),
      input.size(2),
      input.size(3),
      input.size(2)/stride,
      input.size(3),
      stride);
  }));

  //std::cout << output[0] << std::endl;

  return output;
}

std::vector<at::Tensor> shift_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor output,
    at::Tensor xpos,
    at::Tensor ypos,
    const int stride) {
  auto grad_input = at::zeros_like(input);




  const dim3 blocks((input.size(0)*input.size(1)*input.size(2)*input.size(3)+1024-1)/1024);
  const int threads = 1024;

  if(stride==1)
  {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "Shift_Bottom_Backward_Stride1_", ([&] {
      Shift_Bottom_Backward_Stride1<scalar_t><<<blocks, threads>>>(
        grad_output.data<scalar_t>(),
        grad_input.data<scalar_t>(),
        xpos.data<scalar_t>(),
        ypos.data<scalar_t>(),
        input.size(0),
        input.size(1),
        input.size(2),
        input.size(3));
    }));
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "Shift_Bottom_Backward_", ([&] {
      Shift_Bottom_Backward<scalar_t><<<blocks, threads>>>(
        grad_output.data<scalar_t>(),
        grad_input.data<scalar_t>(),
        xpos.data<scalar_t>(),
        ypos.data<scalar_t>(),
        input.size(0),
        input.size(1),
        input.size(2),
        input.size(3));
    }));
  }




  auto grad_xpos_bchw = at::zeros({output.size(0), output.size(1), output.size(2), output.size(3)}, output.options()); // (b,c,h,w)
  auto grad_ypos_bchw = at::zeros({output.size(0), output.size(1), output.size(2), output.size(3)}, output.options()); // (b,c,h,w)

  const dim3 blocks_output((output.size(0)*output.size(1)*output.size(2)*output.size(3)+1024-1)/1024);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "Shift_Position_Backward_", ([&] {
    Shift_Position_Backward<scalar_t><<<blocks_output, threads>>>(
      input.data<scalar_t>(),
      grad_output.data<scalar_t>(),
      grad_input.data<scalar_t>(),
      xpos.data<scalar_t>(),
      ypos.data<scalar_t>(),
      grad_xpos_bchw.data<scalar_t>(),
      grad_ypos_bchw.data<scalar_t>(),
      input.size(0),
      input.size(1),
      input.size(2),
      input.size(3),
      stride);
  }));

  auto grad_xpos_chw = at::mean(grad_xpos_bchw, 0, false);
  auto grad_xpos_ch = at::sum(grad_xpos_chw, 2, false);
  auto grad_xpos_c  = at::sum(grad_xpos_ch, 1, false);
  auto grad_xpos = grad_xpos_c;

  auto grad_ypos_chw = at::mean(grad_ypos_bchw, 0, false);
  auto grad_ypos_ch = at::sum(grad_ypos_chw, 2, false);
  auto grad_ypos_c  = at::sum(grad_ypos_ch, 1, false);
  auto grad_ypos = grad_ypos_c;
  


  const dim3 blocks_norm((output.size(1)+1024-1)/1024);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "applyShiftConstraint_", ([&] {
    applyShiftConstraint<scalar_t><<<blocks_norm, threads>>>(
      grad_xpos.data<scalar_t>(),
      grad_ypos.data<scalar_t>(),
      output.size(1));
  }));

  return {grad_input,grad_xpos,grad_ypos};
}
