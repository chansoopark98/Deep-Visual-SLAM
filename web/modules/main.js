
// import './ar_manager.js';
// import { updateCameraPose } from './ar_manager.js';

let durations_lists = [];

class InferenceManager {
  /**
   * @param {tf.GraphModel} model - TensorFlow.js GraphModel (inference 전용)
   * @param {HTMLCanvasElement} canvas - 카메라 영상이 그려진 canvas 요소
   * @param {Array<number>} inputSize - 모델이 기대하는 [height, width] (예: [224, 224])
   * @param {number} inferenceInterval - 추론 사이의 시간 간격 (ms)
   */
  constructor(model, canvas, inputSize = [224, 224], inferenceInterval = 100) {
    this.model = model;
    this.canvas = canvas;
    this.inputSize = inputSize;
    this.inferenceInterval = inferenceInterval;
    this.prevFrame = null; // 이전 프레임을 저장하여 두 프레임을 이어붙임
    this.isRunning = false;
  }

  // canvas의 현재 프레임을 캡처하고 전처리합니다.
  captureFrame() {
    const startTime = performance.now();  // 시작 시간 기록
    const imgTensor = tf.tidy(() => {
      // tf.browser.fromPixels: canvas를 [height, width, 3] 텐서로 변환
      let tensor = tf.browser.fromPixels(this.canvas);
      // 모델에서 요구하는 크기로 리사이즈 (예: [224, 224])
      tensor = tf.image.resizeNearestNeighbor(tensor, this.inputSize);
      tensor = tensor.toFloat();
      return tensor;
    });
    const endTime = performance.now();    // 종료 시간 기록
    // console.log(`captureFrame 시간: ${endTime - startTime} ms`);
    return imgTensor;
  }

  // 이전 프레임과 현재 프레임을 이어붙여 모델 입력 텐서를 준비합니다.
  prepareInput() {
    const startTime = performance.now();
    // 현재 프레임 캡처
    const currFrame = this.captureFrame(); // shape: [h, w, 3]
    let combined;
    if (this.prevFrame == null) {
      // 첫 실행 시에는 현재 프레임을 두 번 이어붙여 사용
      combined = tf.concat([currFrame, currFrame], -1); // shape: [h, w, 6]
    } else {
      // 이전 프레임과 현재 프레임을 concat
      combined = tf.concat([this.prevFrame, currFrame], -1); // shape: [h, w, 6]
      // 이전 프레임은 사용 후 dispose하여 메모리 해제
      this.prevFrame.dispose();
    }
    // 현재 프레임을 clone하여 다음 추론을 위한 이전 프레임으로 저장
    this.prevFrame = currFrame.clone();
    // 배치 차원 추가: [1, h, w, 6]
    const inputTensor = combined.expandDims(0);
    // combined 텐서는 dispose (inputTensor는 독립적 메모리)
    combined.dispose();
    currFrame.dispose();
    const endTime = performance.now();
    // console.log(`prepareInput 시간: ${endTime - startTime} ms`);
    return inputTensor;
  }

  // 준비된 입력 텐서를 사용하여 모델 추론을 실행합니다.
  async runInference() {
    const startTime = performance.now();
    const inputTensor = this.prepareInput();
    let output;
    try {
      // GraphModel의 경우 executeAsync를 사용하여 비동기로 실행합니다.
      // output = await this.model.executeAsync(inputTensor);
      output = await this.model.execute(inputTensor);
    } catch (error) {
      console.error('Error during inference:', error);
    } finally {
      // 입력 텐서는 더 이상 필요 없으므로 메모리 해제
      inputTensor.dispose();
    }
    const endTime = performance.now();
    // console.log(`runInference 시간: ${endTime - startTime} ms`);
    return output;
  }

  // 주어진 간격으로 반복 추론을 수행합니다.
  // callback(result)는 추론 결과를 처리하는 사용자 정의 함수입니다.
  startContinuousInference(callback) {
    this.isRunning = true;
    const inferLoop = async () => {
      if (!this.isRunning) return;
      const loopStartTime = performance.now();
      const result = await this.runInference();
      if (callback) {
        callback(result);
      }
      const loopEndTime = performance.now();
      durations_lists.push(loopEndTime - loopStartTime);
      console.log(`한 사이클 추론 전체 시간: ${loopEndTime - loopStartTime} ms`);
      // 추론 결과를 사용한 후, 결과 텐서들의 메모리를 관리해 주세요.
      // 예를 들어, 단일 텐서인 경우 result.dispose(), 배열인 경우 각각 dispose()
      setTimeout(inferLoop, this.inferenceInterval);
    };
    inferLoop();
  }

  // 반복 추론을 중지합니다.
  stopContinuousInference() {
    this.isRunning = false;
  }
}

const modelPath = '../assets/tfjs/model.json';
tf.setBackend('webgl');

async function loadAndDisposeModel(url) {
  try {
    const model = await tf.loadGraphModel(url);
    model.dispose(); // 메모리 해제
    return true;
  } catch (error) {
    console.error('모델 로드 실패:', error);
    return false;
  }
}

async function testModelLoad(url, iterations = 100) {
  let successCount = 0;
  let failureCount = 0;

  for (let i = 0; i < iterations; i++) {
    const success = await loadAndDisposeModel(url);
    if (success) {
      successCount++;
    } else {
      failureCount++;
    }
    console.log(`시도 ${i + 1}: 성공 ${successCount}, 실패 ${failureCount}`);
  }

  console.log('테스트 완료');
  console.log(`성공 횟수: ${successCount}`);
  console.log(`실패 횟수: ${failureCount}`);
}

// const modelUrl = 'https://example.com/model/model.json';
await testModelLoad(modelPath);

tf.loadGraphModel(modelPath)
  .then(model => {
    console.log('Model loaded successfully.');
    const canvas = document.getElementById('canvas');
    // InferenceManager 인스턴스 생성 (inputSize는 모델에 맞게 조정)
    const inferenceManager = new InferenceManager(model, canvas, [240, 320], 100);

    // 추론 결과를 처리할 콜백 함수 예시
    const handleResult = result => {
      // console.log('Inference result:', result);
      // 결과 텐서가 단일 텐서인 경우:
      if (result instanceof tf.Tensor) {
        
        result.array().then(matrix => {
          // console.log('Tensor 4x4 matrix:', matrix);
          const rowMajor = matrix.flat();  // 예: [m00, m01, m02, m03, m10, m11, ..., m33]

          // Three.js는 column-major 순서의 배열을 필요로 하므로 재정렬
          const colMajor = [
            rowMajor[0], rowMajor[4], rowMajor[8],  rowMajor[12],
            rowMajor[1], rowMajor[5], rowMajor[9],  rowMajor[13],
            rowMajor[2], rowMajor[6], rowMajor[10], rowMajor[14],
            rowMajor[3], rowMajor[7], rowMajor[11], rowMajor[15]
          ];
          // updateCameraPose(colMajor);
        });
        result.dispose();
        // console.log('Result tensor disposed.');
      }
    };

    // 연속 추론 시작
    inferenceManager.startContinuousInference(handleResult);
  })
  .catch(error => {
    console.error('Error loading the model:', error);
  });
