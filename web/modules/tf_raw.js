// inferenceManager.js
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
      return tf.tidy(() => {
        // tf.browser.fromPixels: canvas를 [height, width, 3] 텐서로 변환
        let imgTensor = tf.browser.fromPixels(this.canvas);
        // 모델에서 요구하는 크기로 리사이즈 (예: [224, 224])
        imgTensor = tf.image.resizeBilinear(imgTensor, this.inputSize);
        // 픽셀 값을 0~255에서 0~1 범위로 정규화
        imgTensor = imgTensor.toFloat().div(tf.scalar(255));
        return imgTensor;
      });
    }
  
    // 이전 프레임과 현재 프레임을 이어붙여 모델 입력 텐서를 준비합니다.
    prepareInput() {
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
      return inputTensor;
    }
  
    // 준비된 입력 텐서를 사용하여 모델 추론을 실행합니다.
    async runInference() {
      const inputTensor = this.prepareInput();
      let output;
      try {
        // GraphModel의 경우 executeAsync를 사용하여 비동기로 실행합니다.
        output = await this.model.execute(inputTensor);
      } catch (error) {
        console.error('Error during inference:', error);
      } finally {
        // 입력 텐서는 더 이상 필요 없으므로 메모리 해제
        inputTensor.dispose();
      }
      return output;
    }
  
    // 주어진 간격으로 반복 추론을 수행합니다.
    // callback(result)는 추론 결과를 처리하는 사용자 정의 함수입니다.
    startContinuousInference(callback) {
        this.isRunning = true;
        const inferLoop = async () => {
          if (!this.isRunning) return;
          const result = await this.runInference();
          if (callback) {
            callback(result);
          }
          // 다음 애니메이션 프레임에서 바로 추론을 이어갑니다.
          requestAnimationFrame(inferLoop);
        };
        inferLoop();
      }
      
  
    // 반복 추론을 중지합니다.
    stopContinuousInference() {
      this.isRunning = false;
    }
  }
  
  // 사용 예시 (예: main.js)
  const modelPath = '../assets/tfjs/model.json';
  tf.setBackend('webgl');

  tf.loadGraphModel(modelPath)
    .then(model => {
      console.log('Model loaded successfully.');
      const canvas = document.getElementById('canvas');
      // InferenceManager 인스턴스 생성 (inputSize는 모델에 맞게 조정)
      const inferenceManager = new InferenceManager(model, canvas, [480, 640], 100);
  
      // 추론 결과를 처리할 콜백 함수 예시
      const handleResult = result => {
        console.log('Inference result:', result);
        // 결과 텐서가 단일 텐서일 경우:
        if (result instanceof tf.Tensor) {
          result.dispose();
        }
        // 또는 여러 텐서가 반환되면 배열 형태로 각 텐서를 dispose해야 합니다.
      };
  
      // 연속 추론 시작
      inferenceManager.startContinuousInference(handleResult);
    })
    .catch(error => {
      console.error('Error loading the model:', error);
    });
  