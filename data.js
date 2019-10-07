function prepareData(data,length,labels) {
  xs = [];

  // Making array of array of each image's pixels 28*28 (i.e. 784)
  // As images are grayscale images each pixel holds values from 0 to 255.
  for (let i = 0; i < length; i++) {
    let offset = i * len;
    xs[i] = data.bytes.subarray(offset, offset + len);
  }

  // Normalising pixel values from (0 to 255) to (0 to 1).
  let newXs = [];
  for (let i = 0; i < xs.length; i++) {
    let xData = xs[i];
    let inputs = Array.from(xData).map(x => x / 255);
    newXs[i] = inputs;
  }

  // preparing target outputs.
  let newLabelsData = [];
  for (var i = 0; i < length; i++) {
    let label = labels[i];
    let targets = [0, 0, 0 , 0 ,0 , 0, 0, 0, 0, 0];
    targets[label] = 1;
    newLabelsData[i] = targets;
  }

  return [tf.tensor2d(newXs).reshape([length, 28, 28, 1]) , tf.tensor2d(newLabelsData)];
}
