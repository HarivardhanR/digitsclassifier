const totalTrainData = 10000;
const totalTestData = 1000;
const len = 784;

let traindigitsData;
let trainlabelsData;

let train_xs;
let train_ys;
let test_xs;
let test_ys;
let guess_xs;

let txt = '';
let etxt = '';
let guessArr = [];

let flag = 0;

let leftBuffer;
let rightBuffer;

let model;

function preload() {
  // Loading data.
  traindigitsData = loadBytes('data/SixtyThousand_TrainDigits.bin');
  trainlabelsData = loadStrings('data/train_labels.txt');
  testdigitsData = loadBytes('data/TenThousand_TestDigits.bin');
  testlabelsData = loadStrings('data/test_labels.txt');
}

function setup() {
  createCanvas(800, 280);
  background(255);

  //creating two canvas
	leftBuffer = createGraphics(280,280);
	rightBuffer = createGraphics(520,280);

  //converting Array of String labels to Integers.
  trainlabelsData = trainlabelsData.map(function(v) {
		return parseInt(v, 10);
	});

  testlabelsData = testlabelsData.map(function(v) {
		return parseInt(v, 10);
	});

  [train_xs,train_ys] = prepareData(traindigitsData,totalTrainData,trainlabelsData);
  [test_xs,test_ys] = prepareData(testdigitsData,totalTestData,testlabelsData);

  let trainButton = select('#train');
  trainButton.mousePressed(function() {
    // flag helps not to train the model while it is training.
    if (flag === 0) {
      flag = 1;
      txt = "Model is training..."
      trainModel().then(function(){
        txt = "Neural Network is Trained with "+totalTrainData+" images";
        flag = 0;
      });
    }
  });

  let testButton = select('#test');
  testButton.mousePressed(function() {
    let percent = test();
	  txt = "Neural Network is tested with "+ totalTestData +" images.\nIt has " +  nf(percent, 2, 2) + "% accuracy.";
  });

  guessArr = [0,0,0,0,0,0,0,0,0,0];
  let guessButton = select('#guess');
  guessButton.mousePressed(function() {
    let inputs = [];
    let img = get(0,0,280,280);
    img.resize(28, 28);
    img.loadPixels();
    for (let i = 0; i < len; i++) {
      let bright = img.pixels[i * 4];
      inputs[i] = (255 - bright) / 255.0;
    }
    let testing = [];
    testing[0] = inputs;
    let xs = tf.tensor2d(testing);
    guess_xs = xs.reshape([1, 28, 28, 1]);
    guess();
    xs.dispose();
    guess_xs.dispose();
  });

  let clearButton = select('#Clear');
  clearButton.mousePressed(function() {
    background(255);
  });

  model = getModel();
}
async function trainModel() {
  return model.fit(train_xs,train_ys, {
    validationData: [test_xs, test_ys],
    epochs: 10,
  });
}
function test(){
  const preds = tf.tidy(()=>{
    return model.predict(test_xs).argMax([-1]);
  });
  const values = preds.dataSync();
  const arr = Array.from(values);
  let count =0;
  for (var i = 0; i < arr.length; i++) {
    if (arr[i] === testlabelsData[i]) {
      count++;
    }
  }
  return 100 * count/totalTestData;
}
function guess() {
  const preds = model.predict(guess_xs);
  const values = preds.dataSync();
  guessArr = Array.from(values);
  let m = max(guessArr);
  etxt = 'You have entered '+ guessArr.indexOf(m);
  preds.dispose();
}
function draw(){
	drawLeftBuffer();
  drawRightBuffer();

  // Paint the off-screen buffers onto the main canvas
  image(leftBuffer, 0, 0);
  image(rightBuffer, 280, 0);
}
function drawLeftBuffer() {
  strokeWeight(8);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}
function drawRightBuffer(){
	rightBuffer.background(218, 247, 166);
  rightBuffer.fill(255);
  rightBuffer.textSize(32);
	for(var i = 0,j=0; i<=280;i+=28,j++){
		if(guessArr[j]<0.01){
			rightBuffer.fill(255);
		}else if(guessArr[j]<0.1){
			rightBuffer.fill(200);
		}else if(guessArr[j]<0.3){
			rightBuffer.fill(150);
		}else if(guessArr[j]<0.6){
			rightBuffer.fill(100);
		}else if(guessArr[j]<0.8){
			rightBuffer.fill(50);
		}else if(guessArr[j]>=0.8){
			rightBuffer.fill(0);
		}
    rightBuffer.ellipse(40,13 + i, 10, 10);
    push();
  	rightBuffer.fill(0);
  	rightBuffer.textSize(10);
  	rightBuffer.text(""+j,47,17+i);
  	pop();
  }
	noStroke();
	rightBuffer.fill(65);
	rightBuffer.textSize(18);
	rightBuffer.text(txt,100,30);
	rightBuffer.text(etxt,100,90);
}
