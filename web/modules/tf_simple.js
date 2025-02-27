// inferenceManager.js

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs'); //tfjs-backend-webgl
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl'); //tfjs-backend-webgl


{/* <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"> </script>
    <!-- <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"></script> -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script> */}

// 메모리 모니터링
setInterval(() => {
    const memoryInfo = tf.memory();
    console.log('Memory state:', {
      numTensors: memoryInfo.numTensors,
      numBytes: memoryInfo.numBytes,
      unreliable: memoryInfo.unreliable
    });
  }, 1000);
  
  class InferenceManager {
    /**
     * @param {tf.GraphModel} model - TensorFlow.js GraphModel (inference 전용)
     * @param {Array<number>} inputSize - 모델이 기대하는 [height, width] (예: [224, 224])
     */
    constructor(model, inputSize = [240, 320]) {
      this.model = model;
      this.inputSize = inputSize;
      this.prevFrame = null; // 이전 프레임을 저장하여 두 프레임을 이어붙임
    }
  
    // canvas의 현재 프레임을 캡처하고 전처리합니다.
    captureFrame(imageData) {
      const startTime = performance.now();  // 시작 시간 기록
      const imgTensor = tf.tidy(() => {
        // tf.browser.fromPixels: canvas를 [height, width, 3] 텐서로 변환
        const { buffer, width, height } = imageData;
        const pixelData = new Uint8ClampedArray(buffer);
        let tensor = tf.browser.fromPixels(new ImageData(pixelData, width, height));
        // 모델에서 요구하는 크기로 리사이즈 (예: [224, 224])
        tensor = tf.image.resizeNearestNeighbor(tensor, this.inputSize);
        tensor = tensor.toFloat();
        return tensor;
      });
      const endTime = performance.now();    // 종료 시간 기록
      console.log(`captureFrame 시간: ${endTime - startTime} ms`);
      return imgTensor;
    }
  
    // 이전 프레임과 현재 프레임을 이어붙여 모델 입력 텐서를 준비합니다.
    prepareInput(imageData) {
      const startTime = performance.now();
      // 현재 프레임 캡처
      const currFrame = this.captureFrame(imageData); // shape: [h, w, 3]
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
      console.log(`prepareInput 시간: ${endTime - startTime} ms`);
      return inputTensor;
    }
  
    // 준비된 입력 텐서를 사용하여 모델 추론을 실행합니다.
    async runInference(imageData) {
      const startTime = performance.now();
      const inputTensor = this.prepareInput(imageData);
      let output;
      try {
        // GraphModel의 경우 executeAsync를 사용하여 비동기로 실행합니다.
        output = await this.model.executeAsync(inputTensor);
      } catch (error) {
        console.error('Error during inference:', error);
      } finally {
        // 입력 텐서는 더 이상 필요 없으므로 메모리 해제
        inputTensor.dispose();
      }
      const endTime = performance.now();
      console.log(`runInference 시간: ${endTime - startTime} ms`);
      return output;
    }
  }

const modelPath = '../assets/tfjs/model.json';
tf.setBackend('webgl');

async function init() {
  const vo_model = await tf.loadGraphModel(modelPath);
  console.log('Model loaded successfully.');
  
  // InferenceManager 인스턴스 생성 (inputSize는 모델에 맞게 조정)
  const inferenceManager = new InferenceManager(vo_model, [240, 320]);
  
  // worker 메시지 처리: 'frame' 타입의 메시지 처리
  onmessage = async (event) => {
    const msg = event.data;
    if (msg.type === 'frame') {
      // msg.data: { buffer, width, height }
      const imageData = msg.data;
      const result = await inferenceManager.runInference(imageData);
      // 추론 결과를 메인 스레드로 전송 (간단한 JSON 직렬화 가능한 형태)
      postMessage({ type: 'result', result });
    }
  };
}
init();
