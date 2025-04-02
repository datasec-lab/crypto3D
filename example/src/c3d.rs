use ::network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};
use std::time::{Duration, Instant};

use super::*;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{
    path::Path,
    io::Read,
};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
    0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
    0x52, 0xd2,
];

pub fn construct_c3d<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        2 => vec![4, 8],             
        4 => vec![2, 4, 7, 9],         
        6 => vec![1, 3, 5, 7, 9, 10],   
        8 => vec![1, 2, 3, 4, 6, 7, 8, 9],
        10 => vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
       // _ => ((10 - num_poly + 1)..=10).collect::<Vec<_>>(),
         _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers:      vec![],
            eval_method: :: network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };

    let input_dims = (batch_size, 3, 774, 112, 112);

    let kernel_dims = (64, 3, 3, 3, 3);
    // sample_conv_layer(vs, input_dims,kernel_dims, stride,padding,rng)
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    //relu activation
    add_activation_layer(&mut network, &relu_layers);

   
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
  
    add_activation_layer(&mut network, &relu_layers);
    
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));
    

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (128, 64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
  
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
 

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 128, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);



    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));

    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 256, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

  
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

  

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));
    


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
   
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2, 2), 2);
    network.layers.push(Layer::LL(pool));
    

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 101, rng);
    network.layers.push(Layer::LL(fc));

    network
}

