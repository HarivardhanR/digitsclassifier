function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // Creating first layer of our convolutional neural network and
  // specifying input shape as well as we have some parameters for
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // Creating MaxPooling layer which acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Adding a flatten layer to flatten the output from the 2D filters into a
  // 1D vector to prepare it for input into our last layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer(full connected) which has 10 output units,
  // one for eachoutput (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUTS = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUTS,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // Choosing optimizer, loss function and accuracy metric,
  // then compile and return the model.
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}
