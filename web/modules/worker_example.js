// inferenceWorker.js

// tf.js 라이브러리 로드 (CDN 등에서 제공하는 파일 경로로 수정)
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

// 모델 로드에 필요한 경로 (환경에 맞게 조정)
const modelPath = '../assets/tfjs/model.json';

// 모델 입력 크기 (예: [height, width])
const workerInputSize = [480, 640];

// 모델과 이전 프레임 저장 변수
let model = null;
let prevFrameTensor = null;

// WebGL 백엔드 설정 후 모델 로드
tf.setBackend('webgl').then(() => {
  tf.loadGraphModel(modelPath)
    .then(m => {
      model = m;
      postMessage({ type: 'status', message: 'Model loaded successfully.' });
    })
    .catch(err => {
      postMessage({ type: 'error', message: 'Error loading model: ' + err });
    });
});

// 캡처된 이미지 데이터를 받아 모델 입력 텐서를 준비합니다.
function prepareInput(imageData) {
  return tf.tidy(() => {
    const { buffer, width, height } = imageData;
    // 복원: transferable로 전달된 ArrayBuffer를 Uint8ClampedArray로 변환
    const pixelData = new Uint8ClampedArray(buffer);
    // tf.browser.fromPixels는 {data, width, height} 형태의 객체를 인자로 받음
    let currFrame = tf.browser.fromPixels({ data: pixelData, width, height });
    // 모델이 기대하는 크기로 리사이즈
    currFrame = tf.image.resizeNearestNeighbor(currFrame, workerInputSize);
    currFrame = currFrame.toFloat();

    // 이전 프레임이 없으면 현재 프레임을 두 번 concat
    let combined;
    if (prevFrameTensor == null) {
      combined = tf.concat([currFrame, currFrame], -1); // shape: [h, w, 6]
    } else {
      combined = tf.concat([prevFrameTensor, currFrame], -1);
      prevFrameTensor.dispose();
    }
    // clone()을 사용하여 현재 프레임을 이전 프레임으로 저장
    prevFrameTensor = currFrame.clone();

    // 배치 차원 추가: [1, h, w, 6]
    const inputTensor = combined.expandDims(0);
    // combined와 currFrame은 tf.tidy()에 의해 자동 dispose됨
    return inputTensor;
  });
}

// 프레임을 받아 추론 실행
async function runInference(imageData) {
  if (!model) {
    console.warn('Model not loaded yet.');
    return;
  }
  const inputTensor = prepareInput(imageData);
  let output;
  try {
    // executeAsync를 통해 비동기로 추론 실행
    output = await model.executeAsync(inputTensor);
  } catch (error) {
    console.error('Error during inference:', error);
    inputTensor.dispose();
    return;
  }
  inputTensor.dispose();

  // output이 tf.Tensor인 경우 결과를 배열로 추출하고 dispose합니다.
  let outputData;
  if (output instanceof tf.Tensor) {
    outputData = await output.data();
    output.dispose();
  } else if (Array.isArray(output)) {
    // 여러 텐서가 반환된 경우 각각 처리
    outputData = [];
    for (const t of output) {
      outputData.push(Array.from(await t.data()));
      t.dispose();
    }
  }
  return outputData;
}

// worker 메시지 처리: 'frame' 타입의 메시지 처리
onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === 'frame') {
    // msg.data: { buffer, width, height }
    const imageData = msg.data;
    const result = await runInference(imageData);
    // 추론 결과를 메인 스레드로 전송 (간단한 JSON 직렬화 가능한 형태)
    postMessage({ type: 'result', result });
  }
};
