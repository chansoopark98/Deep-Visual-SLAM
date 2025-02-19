// inferenceManager.js

const modelPath = '../assets/tfjs/model.json';
tf.setBackend('webgl');

tf.ready().then(() => {
  console.log('Backend is ready.');

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
      this.prevFrame = null; // 이전 프레임 저장
      this.isRunning = false;
      // 매번 생성하는 스칼라 대신 한 번 생성하여 재사용
      this.divisor = tf.scalar(255);
    }
  
    // canvas의 현재 프레임을 캡처하고 전처리합니다.
    captureFrame() {
      return tf.tidy(() => {
        let imgTensor = tf.browser.fromPixels(this.canvas);         // [h, w, 3]
        imgTensor = tf.image.resizeBilinear(imgTensor, this.inputSize); // [inputSize[0], inputSize[1], 3]
        // this.divisor는 재사용하며, tf.tidy 내에서 자동 dispose되지 않습니다.
        const normalized = imgTensor.toFloat().div(this.divisor);
        return normalized.clone(); // clone해서 tidy 종료 후에도 살아있게 함
      });
    }
  
    // 이전 프레임과 현재 프레임을 이어붙여 모델 입력 텐서를 준비합니다.
    prepareInput() {
      // tf.tidy 내에서 모든 임시 텐서를 정리합니다.
      const inputTensor = tf.tidy(() => {
        const currFrame = this.captureFrame(); // shape: [h, w, 3]
        let combined;
        if (this.prevFrame == null) {
          // 첫 실행 시 현재 프레임을 두 번 이어붙임: [h, w, 6]
          combined = tf.concat([currFrame, currFrame], -1);
        } else {
          combined = tf.concat([this.prevFrame, currFrame], -1);
          this.prevFrame.dispose();
        }
        // 현재 프레임을 clone하여 다음 추론을 위한 이전 프레임으로 저장
        this.prevFrame = currFrame.clone();
        // 배치 차원 추가: [1, h, w, 6]
        const batched = combined.expandDims(0);
        // combined는 tf.tidy가 종료되면서 자동 정리됨
        return batched;
      });
      return inputTensor;
    }
  
    // 모델 추론을 실행합니다.
    async runInference() {
      // prepareInput 내의 모든 임시 텐서는 tf.tidy로 정리됨
      const inputTensor = this.prepareInput();
      console.log('prepare input!', inputTensor.shape);
      let output;
      try {
        // executeAsync는 결과 텐서를 반환하며, 이 텐서는 tidy 외부이므로 callback에서 직접 dispose해야 함
        output = await this.model.executeAsync(inputTensor);
      } catch (error) {
        console.error('Error during inference:', error);
      } finally {
        inputTensor.dispose();
      }
      return output;
    }
  
    // 주어진 간격으로 반복 추론을 수행합니다.
    // callback(result)는 추론 결과를 처리한 후 dispose를 수행해야 합니다.
    startContinuousInference(callback) {
      this.isRunning = true;
      const inferLoop = async () => {
        if (!this.isRunning) return;
        const result = await this.runInference();
        // 결과 처리는 콜백 함수에서 한 번만 dispose합니다.
        if (callback) {
          await callback(result);
        }
        setTimeout(inferLoop, this.inferenceInterval);
      };
      inferLoop();
    }
  
    // 반복 추론을 중지하고, 보관 중인 임시 텐서를 모두 정리합니다.
    stopContinuousInference() {
      this.isRunning = false;
      if (this.prevFrame) {
        this.prevFrame.dispose();
        this.prevFrame = null;
      }
      if (this.divisor) {
        this.divisor.dispose();
        this.divisor = null;
      }
    }
  }
  
  tf.loadGraphModel(modelPath)
    .then(model => {
      console.log('Model loaded successfully.');
      const canvas = document.getElementById('canvas');
      // InferenceManager 인스턴스 생성 (inputSize는 모델에 맞게 조정)
      const inferenceManager = new InferenceManager(model, canvas, [480, 640], 100);
  
      // 추론 결과를 처리할 콜백 함수
      // 결과 텐서의 데이터를 읽은 후, 단 한 번 dispose합니다.
      const handleResult = async result => {
        if (result) {
          if (Array.isArray(result)) {
            const dataArr = await Promise.all(result.map(t => t.data()));
            console.log('Inference result data arr:', dataArr);
            result.forEach(t => t.dispose());
          } else {
            const data = await result.data();
            console.log('Inference result data:', data);
            result.dispose();
          }
        }
      };
  
      inferenceManager.startContinuousInference(handleResult);
  
      // 예: 일정 시간 후 추론 중지 (필요 시)
      // setTimeout(() => inferenceManager.stopContinuousInference(), 30000);
    })
    .catch(error => {
      console.error('Error loading the model:', error);
    });
});
